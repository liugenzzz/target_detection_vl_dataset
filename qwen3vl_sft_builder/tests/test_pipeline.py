"""回归测试。不依赖外部数据和 VLM 服务，可以在服务器上直接跑：

    python -m pytest tests/ -q     或     python tests/test_pipeline.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sample import IMAGE_TOKEN, validate_sample
from core.classes import _detect_confusable, _one_char_apart
from core.coords import yolo_to_bbox2d, zone_of
from core.grouping import source_group_key
from core.referring import spatial_phrase
import prompts
from core import phrase_bank


def test_coords_full_image():
    # 全图框应落在 [0,1000] 两端
    assert yolo_to_bbox2d(0.5, 0.5, 1.0, 1.0, 640, 480) == [0, 0, 1000, 1000]


def test_coords_known_value():
    # 像素框 [64,48,576,432] @640x480 -> [100,100,900,900]
    assert yolo_to_bbox2d(0.5, 0.5, 0.8, 0.8, 640, 480) == [100, 100, 900, 900]


def test_coords_scale_invariant():
    """同一个归一化框，不同原图尺寸下 bbox_2d 必须一致 —— 这是 0~1000
    坐标系的核心性质，也是阈值改用面积占比的依据。"""
    a = yolo_to_bbox2d(0.4, 0.6, 0.1, 0.2, 128, 128)
    b = yolo_to_bbox2d(0.4, 0.6, 0.1, 0.2, 2048, 1440)
    assert a == b


def test_coords_origin_shift():
    assert yolo_to_bbox2d(0.5, 0.5, 1.0, 1.0, 640, 480, origin=1) == [1, 1, 1001, 1001]


def test_zone_and_phrase():
    assert zone_of(0.1, 0.1) == "上左"
    assert zone_of(0.5, 0.5) == "中中"
    assert spatial_phrase(0.9, 0.9) == "下方右侧"


def test_grouping_video_frames_and_augments():
    """同一视频的相邻帧、同一原图的不同增强，必须归到同一组。"""
    base = "clip_mp4-{}_jpg.rf.{}.txt"
    keys = {
        source_group_key(base.format(14, "062732212bcc96d202df7b978c5e2987")),
        source_group_key(base.format(15, "b1db71c78b1af7ab43b8e41758441100")),
        source_group_key(base.format(15, "ef83bd806997ea0c925bdfdd24800d9d")),
    }
    assert keys == {"clip"}
    assert source_group_key("voc8_9948.txt") == "voc8_9948"


def test_confusable_detection():
    groups, hyper = _detect_confusable({3: "一般人员", 9: "人员", 18: "军事人员",
                                        23: "切管器", 24: "切管机",
                                        5: "两栖战舰", 6: "主力战舰", 2: "suv"})
    assert 9 in groups and "军事人员" in groups[9]      # 包含关系
    assert 23 in groups and "切管机" in groups[23]      # 一字之差
    assert 5 not in groups                              # 差两字，视觉可区分
    assert _one_char_apart("压接钳", "压管钳")
    assert not _one_char_apart("剪刀", "剪线钳")

    # 包含关系要单独标出来：它多半是上下位词，对【拒答样本】有害 ——
    # 图里有军事人员，问「有没有人员」答「没有」是错的。
    # 一字之差的是并列的不同东西（切管器 vs 切管机），答「没有」才成立。
    assert 9 in hyper and "军事人员" in hyper[9]
    assert 23 not in hyper


def test_hard_negative_never_asks_about_a_hypernym():
    """图里有遮阳三轮车，问「有没有三轮车」答「没有」是错的 ——
    遮阳三轮车本来就是三轮车。上下位词要从【整个】拒答池里排掉，
    不只是难负样本那一路：随机兜底那一路照样会抽到它。"""
    import random
    from core.tasks import Ctx, exist_negative
    ctx = Ctx(annotation=None, boxes=[_Box(0, "遮阳三轮车")], grades={}, vlm={},
              all_labels=["三轮车", "遮阳三轮车", "切管器", "切管机", "直升机"],
              bbox2d=lambda b: [1, 2, 3, 4], spatial=lambda b: "中间",
              rng=random.Random(0), measure_words={"遮阳三轮车": "辆"},
              confusable={"遮阳三轮车": ["三轮车", "切管器"]},
              hypernym={"遮阳三轮车": ["三轮车"]})
    asked = set()
    for i in range(200):
        ctx.rng = random.Random(i)
        out = exist_negative(ctx)
        if out and out["polarity"] == "negative":
            asked.add(out["label"])
    assert asked, "一条拒答样本都没生成"
    assert "三轮车" not in asked, f"上下位词漏进拒答池：{sorted(asked)}"


def _sample(convs):
    return {"id": "t", "images": ["a.jpg"], "conversations": convs}


def test_validate_rejects_duplicate_image_token():
    """<image> 只能出现一次且必须在第一轮 —— 多轮里重复注入会让训练崩。"""
    bad = _sample([
        {"from": "human", "value": f"{IMAGE_TOKEN}\n甲"}, {"from": "gpt", "value": "1"},
        {"from": "human", "value": f"{IMAGE_TOKEN}\n乙"}, {"from": "gpt", "value": "2"},
    ])
    assert any("必须恰好 1 次" in i for i in validate_sample(bad))


def test_validate_rejects_wrong_role_order():
    bad = _sample([{"from": "gpt", "value": f"{IMAGE_TOKEN}\n甲"},
                   {"from": "human", "value": "1"}])
    assert any("from 应为" in i for i in validate_sample(bad))


def test_validate_accepts_good_sample():
    good = _sample([
        {"from": "human", "value": f"{IMAGE_TOKEN}\n图中中部中间那个目标是什么？"},
        {"from": "gpt", "value": "是人员。"},
        {"from": "human", "value": "请给出它在图中的位置。"},
        {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4],"label":"人员"}'},
    ])
    assert validate_sample(good) == []


# ---------------------------------------------------------------------------
# 以下是代码审查发现的缺陷的回归测试。每一条都对应一个真实修复。
# ---------------------------------------------------------------------------

def _client(**over):
    from config import Config
    from core.vlm_client import VlmClient
    cfg = Config({"vlm": dict({"enabled": True, "api_url": "http://x/v1",
                               "cache_dir": ""}, **over)})
    return VlmClient(cfg)


def test_prefetch_survives_missing_cache_dir():
    """审查发现 #1：cache_dir 未配置时，预取结果只写磁盘会被全部丢弃 ——
    白烧一整轮 API，却产出纯模板数据集且不报错。结果必须同时进内存。"""
    c = _client(cache_dir="")            # 故意不配磁盘缓存
    assert c._cache_path(Path("a.jpg"), [640, 480]) is None

    raw = ('{"picked":[{"id":0,"attribute":"银灰色","color":"银灰色",'
           '"description":"一辆银灰色的车停在路边。","questions":["框出那辆车。"]}]}')
    c._memory[c._key(Path("a.jpg"), [640, 480])] = raw    # 内存里存原始文本
    c._prefetch_done = True

    got = c.scene_info(Path("a.jpg"), [640, 480], {0})
    assert got[0]["attribute"] == "银灰色", "内存里的预取结果必须被取用"


def test_assembly_stage_never_sends_requests():
    """审查发现的次生问题：组装阶段是串行的，那里发请求会带着重试和超时
    把整批任务拖垮（10 万条里 1% 预取失败 = 1000 次串行调用）。
    预取没拿到的直接当「这张图没挑中目标」，相关任务跳过。"""
    c = _client(cache_dir="")
    c._prefetch_done = True
    c._post = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("组装阶段不允许发起网络请求"))
    assert c.scene_info(Path("b.jpg"), [640, 480], {0}) == {}
    assert c.raw_result(Path("b.jpg"), [640, 480]) is None


def test_empty_json_is_not_success():
    """形状不对的 JSON 若被当成成功，会写进一条空缓存，之后每次运行都命中它，
    把整张图永久钉死在「VLM 没挑中任何目标」上，而报告里还显示成功。"""
    from core.vlm_client import _parse_scene_json
    assert _parse_scene_json("") is None
    assert _parse_scene_json("模型今天不想干活") is None
    assert _parse_scene_json('{"picked": []}') is None
    assert _parse_scene_json('{"picked": "不是列表"}') is None
    assert _parse_scene_json('{"选中": [{"id": 0}]}') is None        # 键名写错
    ok = _parse_scene_json('{"picked":[{"id":0,"attribute":"银灰色",'
                           '"color":"银灰色","description":"一辆车。",'
                           '"questions":["框出图中银灰色的车。"]}]}')
    assert ok is not None and ok[0]["attribute"] == "银灰色"



def test_windows_path_yaml_error_is_explained():
    """Windows 用户把路径写成 "F:\\AI-Haishi\\..." 时，YAML 双引号会把反斜杠
    当转义符而报 unknown escape character。默认的报错是一堆 yaml 内部堆栈，
    对用户毫无帮助 —— 必须直接告诉他改成单引号。"""
    import tempfile
    from config import _load_yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8",
                                     delete=False) as fh:
        fh.write('paths:\n  labels_dir: "F:\\AI-Haishi\\project\\labels"\n')
        bad = Path(fh.name)
    try:
        _load_yaml(bad)
        raise AssertionError("非法 YAML 应该报错")
    except ValueError as exc:
        text = str(exc)
        assert "Windows" in text, "报错必须点明是 Windows 路径问题"
        assert "单引号" in text, "报错必须给出可照做的解法"
    finally:
        bad.unlink()


def test_windows_path_all_three_styles_parse():
    """三种推荐写法都必须能正确解析出原始路径。"""
    import yaml
    assert yaml.safe_load(r"p: 'F:\AI-Haishi\labels'")["p"] == r"F:\AI-Haishi\labels"
    assert yaml.safe_load('p: "F:/AI-Haishi/labels"')["p"] == "F:/AI-Haishi/labels"
    assert yaml.safe_load(r'p: "F:\\AI-Haishi\\labels"')["p"] == r"F:\AI-Haishi\labels"


def test_missing_path_error_is_actionable():
    """路径没填时，报错要指出填哪个文件、也可以用哪个环境变量。"""
    from config import Config
    try:
        Config({"paths": {"labels_dir": ""}}).require("paths.labels_dir")
        raise AssertionError("应该报错")
    except ValueError as exc:
        assert "LABELS_DIR" in str(exc), "应提示可用的环境变量"



def test_config_errors_are_not_retried():
    """审查/实战发现：401 被重试了 3 次。认证失败重试永远不会成功 ——
    十万条任务会变成三十万次注定失败的请求砸向服务，还要加上指数退避的等待。
    4xx 里只有 429 限流该重试，其余都是配置问题。"""
    from core.vlm_client import _diagnose

    for code in (400, 401, 403, 404, 422):
        assert _diagnose(code, "") is not None, f"HTTP {code} 是配置问题，不该重试"
    for code in (429, 500, 502, 503):
        assert _diagnose(code, "") is None, f"HTTP {code} 是临时故障，应该重试"


def test_config_error_messages_are_actionable():
    """每种配置错误都要给出能直接照做的解法，而不只是报个状态码。"""
    from core.vlm_client import _diagnose

    assert "VLM_API_KEY" in _diagnose(401, ""), "401 要告诉用户怎么设 key"
    assert "/v1/chat/completions" in _diagnose(404, ""), "404 要给出完整路径示例"
    assert "vlm.model" in _diagnose(400, ""), "400 要指向模型名配置"


def test_fatal_error_carries_message():
    """FatalVlmError 必须能带消息 —— 曾因 @dataclass 误装饰到它头上而
    退化成不接受参数，抛错时反而盖掉了真正的提示。"""
    from core.vlm_client import FatalVlmError

    assert str(FatalVlmError("认证失败")) == "认证失败"













def test_all_tasks_share_one_truth_set_per_image():
    """同一张图上，所有任务必须基于同一个目标集合。

    过滤后的框【就是】这张图的真值。曾经用「该类全部框都合格」来决定能否出
    盘点/检测类任务，结果同一张图上一个样本说「有 2 辆卡车」（人员整类因有框
    被过滤而不进清单），另一个样本又去定位人员 —— 上下文自相矛盾。
    Ctx 里不该再有 clean_labels 这类会让部分类别时隐时现的字段。
    """
    import dataclasses
    from core.tasks import Ctx

    fields = {f.name for f in dataclasses.fields(Ctx)}
    assert "clean_labels" not in fields, "不能再按类别可见性分叉，会造成同图矛盾"
    assert "boxes" in fields, "所有任务共用 boxes 这一个集合"


def test_inventory_lists_every_kept_class():
    """盘点清单必须覆盖过滤后剩下的每一个类别，不能漏。
    漏一个类别，同一张图上别的任务去定位它时就会与清单打架。"""
    from pathlib import Path
    from config import load_config
    from core.tasks import Ctx, inventory_locate
    import random

    class _Box:
        def __init__(self, i, label):
            self.index, self.label = i, label
            self.cx = self.cy = 0.5
            self.w = self.h = 0.1

    class _Ann:
        stem = "t"; width = height = 640
        image_path = Path("t.jpg"); label_path = Path("t.txt")

    boxes = [_Box(0, "人员"), _Box(1, "人员"), _Box(2, "卡车")]
    ctx = Ctx(annotation=_Ann(), boxes=boxes, grades={}, 
              vlm={i: {"description": "位于画面左下角，一辆白色卡车停在路边，旁边有树。"}
                   for i in range(3)},
              all_labels=["人员", "卡车", "船"],
              bbox2d=lambda b: [1, 2, 3, 4], spatial=lambda b: "中间",
              rng=random.Random(0), measure_words={"人员": "名", "卡车": "辆"})
    out = inventory_locate(ctx)
    assert out is not None
    inv = dict(item.rsplit("x", 1) for item in out["inventory"])
    assert inv == {"人员": "2", "卡车": "1"}, f"清单不完整：{out['inventory']}"



# ---------------------------------------------------------------- 扩充问法库

def test_bank_rejects_wrong_placeholders():
    req = ["label", "mw"]
    # 少一个占位符 -> 问句失去指向
    assert not phrase_bank.accept("x", "那辆{label}在哪？", req, 30, [])
    # 多一个占位符 -> 构建时 .format() 抛 KeyError，会打断整批
    assert not phrase_bank.accept("x", "那{mw}{label}{color}在哪？", req, 30, [])
    assert phrase_bank.accept("x", "那{mw}{label}在哪？", req, 30, [])


def test_bank_rejects_junk():
    assert not phrase_bank.accept("x", "", [], 30, [])
    assert not phrase_bank.accept("x", "# 这是注释", [], 30, [])
    assert not phrase_bank.accept("x", "图" * 40, [], 30, [])
    assert not phrase_bank.accept("x", "图中有什么？", [], 30, ["图中有什么？"])
    # 落单的大括号在 format 时会炸，必须在入库前拦掉
    assert not phrase_bank.accept("x", "图中有什么}", [], 30, [])


def test_bank_sanitize_strips_model_junk():
    assert phrase_bank.sanitize("1. 图中有什么？") == "图中有什么？"
    assert phrase_bank.sanitize('- "画面里有啥？"') == "画面里有啥？"
    assert phrase_bank.sanitize("  · 看得清的有哪些？ ") == "看得清的有哪些？"


def test_bank_length_counts_placeholder_as_short_word():
    # {label} 实到值是「三轮车」这种短词，按字面 7 个字算会误杀合格短句
    assert phrase_bank.visible_len("那{mw}{label}在哪？") == 8


def test_bank_merges_with_handwritten():
    import prompts
    base = prompts.variants("inv_ask_what")
    try:
        prompts.use_bank({"inv_ask_what": ["图里都有啥？", base[0]]})
        merged = prompts.variants("inv_ask_what")
        # 手写的一条都不能丢，重复的那条不能算两遍
        assert set(base) <= set(merged)
        assert len(merged) == len(base) + 1
        assert len(set(merged)) == len(merged)
    finally:
        prompts.use_bank({})
    assert prompts.variants("inv_ask_what") == base


def test_bank_placeholders_of_reads_from_file():
    import prompts
    assert prompts.placeholders_of("inv_ask_box") == ("label", "mw")
    assert prompts.placeholders_of("inv_ask_what") == ()


def test_bank_roundtrip():
    import tempfile, os
    banks = {"inv_ask_what": ["图里都有啥？", "看得见的有哪些？"]}
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        Path(path).write_text(phrase_bank.dump(banks), encoding="utf-8")
        assert phrase_bank.load(path) == banks
    finally:
        os.unlink(path)
    # 没生成过问法库时构建照常跑
    assert phrase_bank.load(None) == {}
    assert phrase_bank.load("/nonexistent/phrase_banks.yaml") == {}




def test_bank_forbid_words_block_semantic_drift():
    # 占位符完全合法，但意思跑到「问位置」上去了 —— 只能靠禁用词拦
    req, forbid = ["label", "mw"], ["在哪", "坐标"]
    assert not phrase_bank.accept("x", "那{mw}{label}在哪？", req, 30, [], forbid)
    assert not phrase_bank.accept("x", "那{mw}{label}的坐标是多少？", req, 30, [], forbid)
    assert phrase_bank.accept("x", "具体说说这{mw}{label}。", req, 30, [], forbid)


def test_every_pool_passes_its_own_rules():
    """每个问法池手写的那几条，必须过得了它自己的占位符与禁用词校验。
    否则规则和种子互相打架，生成时会把模型往一个空集合里赶。"""
    import prompts, yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "config" / "default.yaml").read_text(encoding="utf-8"))
    pools = cfg["phrase_banks"]["pools"]
    glob = tuple(cfg["phrase_banks"]["forbid_global"])
    assert pools, "config 里一个问法池都没登记"
    assert glob, "全局语体禁用词是空的"
    for name in pools:
        required = prompts.placeholders_of(name)
        forbidden = prompts.forbidden_of(name) + glob
        optional = prompts.optional_group_of(name)
        req_any = prompts.required_any_of(name)
        seeds = prompts.load_variants(name)
        assert forbidden, f"{name} 没写 #! forbid，语义闸是空的"
        for v in seeds:
            cap = prompts.max_len_of(name, 30)
            assert phrase_bank.accept(name, v, required, cap, [], forbidden,
                                      optional, req_any), \
                f"{name} 自己的种子 {v!r} 过不了自己的规则"




def test_bank_optional_group_is_all_or_nothing():
    req, grp = ["label", "mw"], ["label", "mw"]
    # 承接上文，整组省掉 —— 允许
    assert phrase_bank.accept("x", "它周围是什么情况？", req, 30, [], (), grp)
    # 只带一半：剩下「说说这{mw}。」或量词丢了的「说说这三轮车。」，比不提还糟
    assert not phrase_bank.accept("x", "说说这{mw}。", req, 30, [], (), grp)
    assert not phrase_bank.accept("x", "说说这{label}。", req, 30, [], (), grp)
    # 没声明 optional-group 的池子，一个都不能少
    assert not phrase_bank.accept("x", "它在哪？", req, 30, [], (), ())
    # 单成员的组 = 那一个可以独立省略（答案池的 {a}：问句刚点过主体）
    assert phrase_bank.accept("x", "在{b}的{rel}。", ["a", "b", "rel"], 30, [],
                              (), ["a"])
    assert not phrase_bank.accept("x", "{rel}。", ["a", "b", "rel"], 30, [], (), ["a"])


def test_bank_install_drops_bad_lines(tmp_path=None):
    """问法库是给人手改的，改坏了不能带着跑 —— install 会再校验一遍。"""
    import tempfile, os, prompts
    from config import Config
    banks = {"ask_describe": [
        "具体说说这{mw}{label}。",          # 好
        "给出那{mw}{label}的坐标。",        # 禁用词：跑到「要坐标」上去了
        "描述一下这{label}。",              # 占位符掉了一半
        "具体说说这{mw}{label}。",          # 重复
    ]}
    fd, path = tempfile.mkstemp(suffix=".yaml"); os.close(fd)
    try:
        Path(path).write_text(phrase_bank.dump(banks), encoding="utf-8")
        cfg = Config({"phrase_banks": {"path": path, "max_len": 30}})
        stats = phrase_bank.install(cfg)
        assert stats == {"loaded": 1, "dropped": 3}, stats
        assert "具体说说这{mw}{label}。" in prompts.variants("ask_describe")
        assert "那{mw}{label}在哪？" not in prompts.variants("ask_describe")
    finally:
        os.unlink(path); prompts.use_bank({})




def test_bank_require_any_keeps_detect_class_exhaustive():
    """detect_class 与 ground_unique 占位符相同，只差「单个 vs 全部」的语义。
    少了穷举词就是同一个问句配两种答案，模型只能学成随机猜给一个还是给全部。"""
    req_any = ["所有", "全部"]
    assert phrase_bank.accept("x", "框出图中所有的{label}。", ["label"], 30, [],
                              (), (), req_any)
    assert not phrase_bank.accept("x", "框出图中的{label}。", ["label"], 30, [],
                                  (), (), req_any)




def test_bank_rejects_model_meta_talk():
    """剥掉序号后的元话语结构上完全合法，只能靠句法与元话语词拦。
    实测「3. 这条带了序号」被剥成「这条带了序号」，当成第二轮问句用了。"""
    assert not phrase_bank.accept("x", "这条带了序号", [], 30, [])
    assert not phrase_bank.accept("x", "以下是几种写法：", [], 30, [])
    assert not phrase_bank.accept("x", "例如可以这样问。", [], 30, [])
    # 没有句末标点 = 多半不是一句完整的问话
    assert not phrase_bank.accept("x", "框出图中的车", [], 30, [])
    assert phrase_bank.accept("x", "框出图中的车。", [], 30, [])
    assert phrase_bank.accept("x", "车在什么位置？", [], 30, [])




class _Box:
    """最小的框替身：任务函数只用到 index / label / cx / cy。"""
    def __init__(self, i, label, cx=0.5, cy=0.5):
        self.index, self.label = i, label
        self.cx, self.cy = cx, cy
        self.w = self.h = 0.1


def _ctx_with_filtered(task_boxes, raw_counts, **kw):
    """构造一个「有框被质量过滤掉了」的 Ctx：boxes 是过滤后的，raw_counts 是过滤前的。"""
    import random
    from core.tasks import Ctx
    return Ctx(annotation=None, boxes=task_boxes, grades={},
               vlm={b.index: {"description": "位于画面左下角，一辆银灰色的卡车停在路边，"
                                             "旁边有一棵树。"} for b in task_boxes},
               all_labels=["人员", "卡车", "船"], bbox2d=lambda b: [1, 2, 3, 4],
               spatial=lambda b: "中间", rng=random.Random(0),
               measure_words={"人员": "名", "卡车": "辆"},
               raw_counts=raw_counts, **kw)


def test_exhaustive_questions_respect_filtered_boxes():
    """穷举式问句必须按【原始标注】把关，不能按过滤后的框把关。

    原图 4 辆三轮车、3 辆太小被过滤，仍问「定位图中的三轮车」并只给 1 个框，
    等于在教模型漏检 —— 实测这种情况占 ground_unique 可选组合的 45.2%。
    """
    from core.tasks import ground_unique, detect_class
    B = _Box

    # 卡车过滤后剩 1 个，但原始标注里有 3 个 -> 不能出 ground_unique
    ctx = _ctx_with_filtered([B(0, "卡车")], {"卡车": 3})
    assert ground_unique(ctx) is None
    # 原始标注里也只有 1 个 -> 可以出
    ctx = _ctx_with_filtered([B(0, "卡车")], {"卡车": 1})
    assert ground_unique(ctx) is not None

    # 人员过滤后剩 2 个，原始 5 个 -> 「框出图中所有的人员」答案不完整，不能出
    ctx = _ctx_with_filtered([B(0, "人员"), B(1, "人员")], {"人员": 5})
    assert detect_class(ctx) is None
    ctx = _ctx_with_filtered([B(0, "人员"), B(1, "人员")], {"人员": 2})
    assert detect_class(ctx) is not None


def test_exist_answer_does_not_undercount():
    """该类有实例被过滤掉时，「有 1 名人员」在图里站着 5 个人的情况下就是错的。"""
    from core.tasks import exist_negative
    import random
    ctx = _ctx_with_filtered([_Box(0, "人员")], {"人员": 5})
    ctx.rng = random.Random(1)
    for _ in range(30):
        out = exist_negative(ctx)
        if out and out["polarity"] == "positive":
            answer = out["conversations"][1]["value"]
            assert "1" not in answer, f"报了个偏小的数：{answer}"
            assert answer.startswith("有")
            break
    else:
        raise AssertionError("30 次都没抽到 positive，随机性有问题")




# --------------------------------------------------------------- 语体（指令 vs 聊天）

def test_register_flags_chat_tone():
    from core import register
    forbid = ["诶", "帮我", "呗", "哪儿"]
    assert register.is_instruction("框出图中的卡车。", forbid)
    assert register.is_instruction("卡车在图中的什么位置？", forbid)
    assert not register.is_instruction("诶那辆卡车在哪儿？", forbid)
    assert not register.is_instruction("帮我把人员框出来。", forbid)
    assert not register.is_instruction("以下是几种写法：", forbid)
    assert not register.is_instruction("框出图中的卡车", forbid)   # 句末缺标点


def test_every_question_prompt_is_registered_as_a_pool():
    """tasks.py 里每一个用作【人类问话】的提示词都必须登记进 phrase_banks.pools。

    没登记的就是一句写死的话，会在十万条样本里原样重复上万遍，而且绕开了
    语体闸和扩充问法库。这项测试防的是「新加了个任务，忘了登记问法池」。
    """
    import re as _re, yaml
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "default.yaml").read_text(encoding="utf-8"))
    pools = set(cfg["phrase_banks"]["pools"])
    src = (root / "core" / "tasks.py").read_text(encoding="utf-8")
    used = set(_re.findall(r'render_choice\("(\w+)"', src))
    # 只作答案、不作问话的池子也走同一套校验，一并登记即可
    assert used <= pools, f"这些问法池没登记进 config：{sorted(used - pools)}"
    assert not _re.findall(r'prompts\.render\("(?!vlm_|measure|gen_|review)', src), \
        "tasks.py 里还有写死的单模板问话，应改成 render_choice 并登记问法池"


def test_validate_sample_rejects_chat_question():
    """落盘前的兜底闸：前两道都可能被绕过，这一道扫的是最终产物。"""
    sample = {"images": ["a.jpg"], "conversations": [
        {"from": "human", "value": "<image>\n诶那辆卡车在哪儿？"},
        {"from": "gpt", "value": "{}"}]}
    issues = validate_sample(sample, ["诶", "哪儿"])
    assert any("指令口吻" in i for i in issues), issues
    # 不传禁用词时行为不变（老调用点不受影响）
    assert not validate_sample(sample)


def test_vlm_questions_pass_through_register_gate():
    """ground_attribute 的问句是 VLM 按图现场生成的，不走问法库。
    提示词里要求了指令式，但那是「请它别这么写」，不是保证。"""
    from core.tasks import TASKS
    ground_attribute = TASKS["ground_full"]
    import random
    ctx = _ctx_with_filtered([_Box(0, "卡车")], {"卡车": 1})
    ctx.forbid_chat = ("诶", "帮我", "哪儿")
    ctx.vlm[0].update({"attribute": "银灰色", "color": "银灰色",
                       "describe_kind": "full", "describe_q": "描述这辆卡车。",
                       "description": "一辆银灰色的卡车，停在路边，旁边有棵树。",
                       "questions": ["诶那辆卡车在哪儿？", "帮我框一下卡车。"]})
    out = ground_attribute(ctx)
    assert out is not None
    assert out["question_source"] == "template", "闲聊问句应被丢弃并回落模板"
    ctx.vlm[0]["questions"] = ["输出银灰色卡车的检测框。"]
    out = ground_attribute(ctx)
    assert out["question_source"] == "vlm"
    assert out["conversations"][0]["value"].endswith("输出银灰色卡车的检测框。")




# --------------------------------------------------------- 跨任务一致性

def _claim_sample(task, answers, **meta):
    convs = []
    for i, a in enumerate(answers):
        convs.append({"from": "human", "value": ("<image>\n问。" if i == 0 else "问。")})
        convs.append({"from": "gpt", "value": a})
    return {"images": ["a.jpg"], "conversations": convs,
            "metadata": dict(task_type=task, **meta)}


def test_consistency_catches_contradictions():
    """核对必须真的能抓到冲突，否则 0 violations 只是空转。"""
    from core import consistency
    truth = {"a.jpg": {"人员": 3}}

    # 盘点说 3 名人员，detect_class 却给 2 个框
    bad = [_claim_sample("inventory_locate", ["有3名人员。"], inventory=["人员x3"]),
           _claim_sample("detect_class", ['[{"bbox_2d":[1,2,3,4],"label":"人员"},'
                                    '{"bbox_2d":[5,6,7,8],"label":"人员"}]'],
                   label="人员", n_boxes=2)]
    out = consistency.check(bad, truth)
    assert out["violations"], "数量说法不一致没被抓到"

    # 一条说「没有直升机」，另一条却框出直升机
    bad2 = [_claim_sample("exist_negative", ["没有，图中不存在直升机。"],
                    label="直升机", polarity="negative"),
            _claim_sample("ground_unique", ['{"bbox_2d":[1,2,3,4],"label":"直升机"}'])]
    out = consistency.check(bad2, {})
    assert any("却框出" in v for v in out["violations"]), out["violations"]

    # 说的框数超过过滤后实际有的
    bad3 = [_claim_sample("detect_class", ['[{"bbox_2d":[1,2,3,4],"label":"人员"},'
                                     '{"bbox_2d":[5,6,7,8],"label":"人员"}]'],
                    label="人员", n_boxes=2)]
    assert consistency.check(bad3, {"a.jpg": {"人员": 1}})["violations"]

    # 对得上的不该报
    good = [_claim_sample("inventory_locate", ["有3名人员。"], inventory=["人员x3"]),
            _claim_sample("detect_class", ['[{"bbox_2d":[1,2,3,4],"label":"人员"},'
                                     '{"bbox_2d":[5,6,7,8],"label":"人员"},'
                                     '{"bbox_2d":[9,9,9,9],"label":"人员"}]'],
                    label="人员", n_boxes=3)]
    assert consistency.check(good, truth)["violations"] == []




# --------------------------------------------------------------- 模型池

def test_endpoint_pool_parsing_and_compat():
    from config import Config
    from core.vlm_client import VlmClient
    # 老写法：平铺的 api_url/model，等价于只有一路
    one = VlmClient(Config({"vlm": {"api_url": "http://a/v1/c", "model": "m1",
                                    "concurrency": 4}}))
    assert len(one.endpoints) == 1 and one.concurrency == 4
    # 池子：总并发是各路之和，api_key 不写就继承外层
    pool = VlmClient(Config({"vlm": {"api_key": "k", "endpoints": [
        {"api_url": "http://a/v1/c", "model": "m1", "concurrency": 8},
        {"api_url": "http://b/v1/c", "model": "m2", "concurrency": 4, "name": "备用"}]}}))
    assert pool.concurrency == 12
    assert [e.model for e in pool.endpoints] == ["m1", "m2"]
    assert all(e.api_key == "k" for e in pool.endpoints)
    assert pool.endpoints[1].name == "备用"


def test_endpoint_pool_round_robin_skips_dead():
    """一路配错（401）只摘除那一路，其余照跑 —— 否则一路配错停掉整批。"""
    from config import Config
    from core.vlm_client import VlmClient
    pool = VlmClient(Config({"vlm": {"endpoints": [
        {"api_url": "http://a/v1/c", "model": "m1", "name": "a"},
        {"api_url": "http://b/v1/c", "model": "m2", "name": "b"}]}}))
    assert {pool._pick().name for _ in range(8)} == {"a", "b"}
    pool.endpoints[0].fatal = "HTTP 401 认证失败"
    assert {pool._pick().name for _ in range(4)} == {"b"}
    pool.endpoints[1].fatal = "HTTP 404 路径不对"
    assert pool._pick() is None




def test_endpoint_pool_weights_by_concurrency():
    """一路写 8、一路写 3，说明前者扛得住的量是后者的两倍多。
    均分会把慢的那路压垮、快的那路闲着。"""
    from collections import Counter
    from config import Config
    from core.vlm_client import VlmClient
    pool = VlmClient(Config({"vlm": {"endpoints": [
        {"api_url": "http://a/v1/c", "model": "m1", "concurrency": 8, "name": "快"},
        {"api_url": "http://b/v1/c", "model": "m2", "concurrency": 2, "name": "慢"}]}}))
    got = Counter(pool._pick().name for _ in range(100))
    assert got["快"] == 80 and got["慢"] == 20, got




# --------------------------------------------------------------- 全量质检

def test_review_score_is_the_minimum_not_the_average():
    """一条描述编造了参照物（grounded=1）但问句写得漂亮（instruction=5），
    取平均还有 3.5 分，照样进训练集。质检要看短板。"""
    from core import review
    got = review.parse('{"reviews":[{"id":0,"correct":5,"grounded":1,'
                       '"clear":4,"instruction":5,"issue":"编了个白色轿车"}]}', 1)
    assert got[0]["score"] == 1, got
    passed, reason = review.verdict(got[0], min_score=3, min_dim={"grounded": 3})
    assert not passed and "grounded" in reason


def test_review_parse_rejects_garbage():
    """解析失败必须返回 None —— 当成满分放行等于质检形同虚设。"""
    from core import review
    assert review.parse("", 3) is None
    assert review.parse("模型今天不想干活", 3) is None
    assert review.parse('{"reviews": "不是列表"}', 3) is None
    assert review.parse('{"reviews":[{"id":0}]}', 3) is None      # 一个维度都没给
    # 编了个不存在的编号
    assert review.parse('{"reviews":[{"id":99,"correct":5,"grounded":5,'
                        '"clear":5,"instruction":5}]}', 3) is None
    # 分数超出 1~5 要夹回去，不能让 99 分冲高均值
    got = review.parse('{"reviews":[{"id":0,"correct":99,"grounded":0,'
                       '"clear":3,"instruction":3}]}', 1)
    assert got[0]["correct"] == 5 and got[0]["grounded"] == 1


def test_review_groups_by_image():
    """一张图上的样本必须合并成一次调用 —— 图片 base64 是请求里最大的一块，
    分开发等于把同一张图传 N 遍。十万条样本差的是十万次调用和一万多次调用。"""
    from core import review
    rows = [{"images": ["a.jpg"], "conversations": []},
            {"images": ["a.jpg"], "conversations": []},
            {"images": ["b.jpg"], "conversations": []}]
    g = review.group_by_image(rows)
    assert len(g) == 2 and len(g["a.jpg"]) == 2


def test_review_resolves_bare_filenames(tmp_path=None):
    """images 字段默认存裸文件名（LLaMA-Factory 风格），质检要读图得拼回去。"""
    import tempfile, os
    from core import review
    d = Path(tempfile.mkdtemp())
    (d / "a.jpg").write_bytes(b"x")
    assert review.resolve_image("a.jpg", d) == d / "a.jpg"
    assert review.resolve_image("别处/a.jpg", d) == d / "a.jpg"
    assert review.resolve_image("不存在.jpg", d) is None
    os.unlink(d / "a.jpg"); os.rmdir(d)


def test_review_cache_is_isolated_per_role():
    """质检的缓存键是问答对的哈希，构建用的是 bbox —— 同一个键空间会撞。
    分子目录后，换质检模型只想重跑质检，删一个目录就行。"""
    import shutil, tempfile
    from config import Config
    from core.vlm_client import VlmClient
    # 用临时目录：VlmClient 会真的把 cache_dir 建出来，写死 "/tmp/_c" 在
    # Windows 上会在 C:\ 根下留垃圾目录。路径也要按 Path 比，不能按字符串 ——
    # Windows 的分隔符是反斜杠，字符串比较必然失败。
    root = Path(tempfile.mkdtemp())
    try:
        cfg = Config({"vlm": {"api_url": "http://a/v1/c", "model": "m",
                              "cache_dir": str(root)}})
        assert VlmClient(cfg).cache_dir == root
        assert VlmClient(cfg, role="review").cache_dir == root / "review"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_review_summary_reports_by_task():
    from core import review
    rows = [{"metadata": {"task_type": "ground_unique"},
             "review": {"correct": 5, "grounded": 5, "clear": 4,
                        "instruction": 5, "score": 4, "passed": True}},
            {"metadata": {"task_type": "ground_unique"},
             "review": {"correct": 2, "grounded": 5, "clear": 4,
                        "instruction": 5, "score": 2, "passed": False,
                        "reason": "correct=2 低于下限 3"}},
            {"metadata": {"task_type": "detect_class"}}]      # 没判上的
    out = review.summarize(rows)
    assert out["scored"] == 2 and out["unscored"] == 1
    assert out["pass_rate"] == 0.5
    assert out["by_task"]["ground_unique"]["n"] == 2
    assert out["top_reject_reasons"] == {"correct=2 低于下限 3": 1}




# --------------------------------------------------------- 客观体检指标

def test_chair_counts_only_labels_in_the_table():
    """CHAIR 只统计类别表里的词。描述里的「斑马线」「路灯杆」不在表中，
    无从判定真假 —— 不计入。宁可低估幻觉率，也不能让客观指标出现假阳性。"""
    from core import stats
    labels = ["卡车", "人员", "直升机"]
    rows = [{"images": ["a.jpg"], "conversations": [
        {"from": "human", "value": "问。"},
        {"from": "gpt", "value": "一辆卡车停在斑马线旁，旁边有一架直升机。"}]}]
    out = stats.chair(rows, {"a.jpg": {"卡车"}}, labels)
    # 提到卡车（对）和直升机（错）；斑马线不在表里，不算
    assert out["mentions_total"] == 2 and out["mentions_wrong"] == 1
    assert out["chair_i"] == 0.5 and out["chair_s"] == 1.0
    assert out["top_hallucinated"] == {"直升机": 1}


def test_chair_ignores_box_answers():
    """框答案里的 label 直接来自标注文件，不是模型生成的，不该算进幻觉。"""
    from core import stats
    rows = [{"images": ["a.jpg"], "conversations": [
        {"from": "human", "value": "问。"},
        {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4],"label":"直升机"}'}]}]
    assert stats.chair(rows, {"a.jpg": {"卡车"}}, ["卡车", "直升机"])[
        "mentions_total"] == 0


def test_distinct_n_is_comparable_across_corpus_sizes():
    """原始 Distinct-n 有语料规模偏差：语料越大分母涨得越快，
    1 千条和 10 万条算出来必然是后者低 —— 那是规模造成的不是多样性下降。
    固定子样本量之后跨版本才可比。"""
    from core import stats
    base = [f"这是第{i}条不一样的描述语句。" for i in range(500)]
    small = stats.distinct_n(base, 2)
    big = stats.distinct_n(base * 20, 2)          # 同样的内容，语料大 20 倍
    assert abs(small - big) < 0.15, f"规模一变数就崩了：{small} vs {big}"


def test_coverage_gini_flags_long_tail():
    from core import stats
    even = [{"metadata": {"label": l}} for l in ["a", "b", "c"] * 10]
    skew = [{"metadata": {"label": "a"}}] * 29 + [{"metadata": {"label": "b"}}]
    assert stats.coverage(even, ["a", "b", "c"])["gini"] < 0.05
    assert stats.coverage(skew, ["a", "b", "c"])["gini"] > 0.4
    assert stats.coverage(skew, ["a", "b", "c"])["never_used"] == ["c"]


def test_needs_image_does_not_drag_down_the_core_score():
    """needs_image 衡量的是「有没有训练价值」，不是「对不对」。
    一条不看图也能答对的拒答样本并没有错，不该把综合分拉到 1。"""
    from core import review
    got = review.parse('{"reviews":[{"id":0,"correct":5,"grounded":5,"clear":5,'
                       '"instruction":5,"needs_image":1}]}', 1)
    assert got[0]["score"] == 5, got
    # 但可以用单维度下限单独卡它
    assert not review.verdict(got[0], 3, {"needs_image": 2})[0]







def test_prompt_names_are_globally_unique():
    """提示词按【文件名】加载，与所在目录无关。重名会取到哪一个是不确定的 ——
    改了另一个却不生效，是最难查的那种问题。加载时就要报错。"""
    import prompts
    idx = prompts._index()
    assert len(idx) >= 20
    stems = [p.stem for p in prompts.PROMPT_DIR.rglob("*.txt")]
    assert len(stems) == len(set(stems)), "提示词重名"


def test_every_task_has_its_own_prompt_dir():
    """每个任务的提示词单独成目录，改一个任务不会误伤别的。
    共用件放 _ 开头的目录，目录名上就标出「动它会影响多个任务」。"""
    import prompts
    from core.tasks import TASKS
    dirs = {d.name for d in prompts.PROMPT_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("__")}
    # 七个 ground_<子类型> 是同一条链的变体，提示词在 prompts/describe/<子类型>.txt，
    # 不各占一个目录 —— 它们共用「指代 -> 框」那两轮，只有描述那一轮不同。
    kinds = {f"ground_{k}" for k in __import__("core.tasks", fromlist=["x"]).DESCRIBE_KINDS}
    missing = set(TASKS) - dirs - kinds
    assert not missing, f"这些任务没有自己的提示词目录：{sorted(missing)}"
    for k in kinds:
        f = prompts.PROMPT_DIR / "describe" / f"{k[len('ground_'):]}.txt"
        assert f.exists(), f"{k} 缺提示词文件 {f}"
    shared = {d for d in dirs if d.startswith("_")}
    assert shared == {"_shared", "_vlm", "_tools"}, sorted(shared)




def test_output_options_are_honoured():
    """output.image_path_style / include_metadata 这两个配置项曾经在清理死代码时
    被弄丢过 —— 配置文件里还写着，但没有任何代码在读，改了不生效也不报错。"""
    from core.pipeline import _image_value, _write_jsonl
    import json, tempfile, os
    img = Path("data/images/a.jpg")
    assert _image_value(img, "filename") == "a.jpg"
    assert _image_value(img, "absolute") == str(img.resolve())
    # relative 要把分隔符统一成正斜杠：Windows 上生成、Linux 上训练时，
    # 反斜杠会被当成转义符
    assert "\\" not in _image_value(img, "relative")

    rows = [{"id": "x", "images": ["a.jpg"], "conversations": [],
             "metadata": {"task_type": "t"}}]
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    try:
        _write_jsonl(Path(path), rows, include_meta=False)
        got = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "metadata" not in got and got["id"] == "x"
        # 摘 metadata 不能改坏传进来的对象 —— 后面还要用它算配比和一致性
        assert "metadata" in rows[0]
        _write_jsonl(Path(path), rows, include_meta=True)
        assert "metadata" in json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        os.unlink(path)




# ------------------------------------------------- 问答要对得上、不能宽泛

def test_box_answer_questions_must_ask_for_a_box():
    """答案是坐标的问句必须明说要框/坐标。「这辆自行车在哪？」既可以答
    「在画面左下角」也可以答一个 bbox —— 两种答案都成立，而训练数据只给了
    其中一种，等于在教模型猜。"""
    import prompts
    from core import phrase_bank
    for name in ("ground_unique", "inv_ask_box", "detect_class", "ground_attribute"):
        req = prompts.required_any_of(name)
        forbid = prompts.forbidden_of(name)
        assert req, f"{name} 没声明 require-any，宽泛问法拦不住"
        for w in ("在哪", "什么位置"):
            assert w in forbid, f"{name} 的 forbid 里缺「{w}」"
        for v in prompts.load_variants(name):
            assert any(k in v for k in req), f"{name} 的种子没说要框：{v}"
            assert not any(w in v for w in forbid), f"{name} 的种子命中禁用词：{v}"
        # 宽泛问法要被拦下
        assert not phrase_bank.accept(name, "那辆车在哪？", prompts.placeholders_of(name),
                                      30, [], forbid, prompts.optional_group_of(name), req)


def test_describe_question_names_what_it_wants():
    """答案里有外观、方位、周边三样，问句就得把这三样点出来。
    「介绍一下这辆自行车」——中文里「介绍」是说牌子型号价格，问的和答的
    不是一回事；「描述该区域的内容」太空泛，答成什么样都算对。"""
    import prompts
    from core.tasks import _WANTS_LOOK, _WANTS_WHERE
    assert "介绍" in prompts.forbidden_of("ask_describe")
    for v in prompts.load_variants("ask_describe"):
        assert any(w in v for w in _WANTS_WHERE), f"没点明要方位/周边：{v}"
        assert any(w in v for w in _WANTS_LOOK), f"没点明要外观：{v}"


def test_model_written_describe_question_is_gated():
    """模型和 description 成对写的问句优先用 —— 同一次调用写出来的，问答对得上。
    但它写飘了（宽泛、跑去问坐标、太短）就得回落模板池，不能直接进数据。"""
    from core.tasks import _describe_question
    ctx = _ctx_with_filtered([_Box(0, "卡车")], {"卡车": 1})
    ctx.forbid_chat = ("诶", "帮我")

    good = "描述这辆卡车的外观、所在方位，以及它旁边有什么。"
    ctx.vlm[0]["describe_q"] = good
    assert _describe_question(ctx, ctx.boxes[0]) == good

    for bad in ("介绍一下这辆卡车。",          # 「介绍」问的不是外观方位
                "这辆卡车在哪？",              # 宽泛，而且答案不是坐标
                "给出这辆卡车的坐标。",        # 跑去问坐标了，答案却是描述
                "描述。",                      # 太短
                "帮我说说这辆卡车呗。"):       # 闲聊腔
        ctx.vlm[0]["describe_q"] = bad
        got = _describe_question(ctx, ctx.boxes[0])
        assert got != bad, f"这句本该被拦下：{bad}"
        assert got in [v.format(mw="辆", label="卡车")
                       for v in __import__("prompts").load_variants("ask_describe")]




def test_json_fence_option_and_consistency_still_parses():
    """Qwen3-VL 基座输出坐标时自带 ```json 包裹。数据可以跟着包、也可以不包，
    但一致性核对必须两种都读得懂 —— 否则开了包裹之后核对静默失效，
    永远报 0 冲突。"""
    import json, random
    from core.tasks import _j, ground_unique
    from core import consistency
    assert _j({"a": 1}) == '{"a":1}'
    assert _j({"a": 1}, fence=True).startswith("```json")

    ctx = _ctx_with_filtered([_Box(0, "卡车")], {"卡车": 1})
    ctx.json_fence = True
    out = ground_unique(ctx)
    answer = out["conversations"][1]["value"]
    assert answer.startswith("```json") and answer.rstrip().endswith("```")
    assert json.loads(answer.strip().strip("`").removeprefix("json"))["label"] == "卡车"

    # 一致性核对剥得掉包裹
    sample = {"images": ["a.jpg"], "metadata": {"task_type": "ground_unique"},
              "conversations": [{"from": "human", "value": "问。"},
                                {"from": "gpt", "value": answer}]}
    assert consistency.claims_of(sample)["boxes"], "带包裹的框答案没被解析出来"




# --------------------------------------------------------- 像素核对颜色

def test_color_word_extraction():
    from core import colorcheck
    assert colorcheck.color_word("银灰色") == "银"
    assert colorcheck.color_word("深蓝色") == "蓝"
    assert colorcheck.color_word("白色车身") == "白"
    # 非颜色属性不归这道闸管
    assert colorcheck.color_word("车头朝左") is None
    assert colorcheck.color_word("停在路边") is None
    assert colorcheck.color_word("") is None


def test_color_conflict_catches_the_real_failure():
    """真实踩到的：数据说「框出图中白色车身的卡车」，框却落在一辆蓝色卡车上。
    指代指向了错误的目标，而答案是框 —— 模型会学到「白色」对应蓝色卡车。"""
    from core import colorcheck
    blue = (219, 0.85, 0.78)        # 实测的蓝色卡车
    assert colorcheck.conflicts("白色车身", blue)
    assert colorcheck.conflicts("银灰色", blue)
    assert not colorcheck.conflicts("蓝色", blue)
    # 非颜色属性一律放行
    assert not colorcheck.conflicts("车头朝左", blue)


def test_color_check_is_deliberately_lenient():
    """航拍小目标、阴影、JPEG 压缩都会让颜色发飘。这道闸的定位是
    「拦住离谱的」，不是「精确判定颜色」—— 拿不准一律放行，
    否则会把大量正确样本误杀。"""
    from core import colorcheck
    # 拿不到像素 -> 放行
    assert not colorcheck.conflicts("白色", None)
    # 低饱和的灰白目标，说白说银都放行
    pale = (60, 0.06, 0.88)
    assert not colorcheck.conflicts("白色", pale)
    assert not colorcheck.conflicts("银灰色", pale)
    # 但在这种目标上说「红色」站不住
    assert colorcheck.conflicts("红色", pale)


def test_color_gate_only_drops_the_colour_not_the_target():
    """模型可能颜色看错但位置、类别都对 —— 那个目标换个非颜色属性照样能用。
    只摘颜色说法，不整条丢。"""
    import sys, tempfile
    from pathlib import Path as P
    from PIL import Image
    from core.pipeline import _drop_bad_colors
    from collections import Counter

    d = P(tempfile.mkdtemp()); img = d / "a.jpg"
    im = Image.new("RGB", (200, 200), (150, 150, 150))
    for x in range(60, 140):
        for y in range(60, 140):
            im.putpixel((x, y), (30, 90, 200))     # 蓝色目标
    im.save(img)

    class _Ann:
        image_path = img; width = height = 200
    box = _Box(0, "卡车", cx=0.5, cy=0.5); box.w = box.h = 0.4
    info = {0: {"color": "白色", "attribute": "白色车身",
                "description": "一辆卡车。", "questions": ["框出卡车。"]}}
    stats = Counter()
    _drop_bad_colors(_Ann(), [box], info, stats, [])
    assert info[0]["color"] == "" and info[0]["attribute"] == ""
    assert stats["dropped_color"] == 1 and stats["dropped_attribute"] == 1
    # 其余字段原样保留 —— 目标本身没问题
    assert info[0]["description"] == "一辆卡车。"
    assert info[0]["questions"] == ["框出卡车。"]




def test_attribute_must_identify_uniquely_among_same_class():
    """真实踩到：一张公园图里好几个人都穿白上衣，VLM 给其中两个都写了
    「穿白色上衣」，于是生成了两条问句几乎一样、答案却是不同框的样本 ——
    同一个问题两个正确答案，模型只能学成随机猜。"""
    import random
    from core.tasks import Ctx, TASKS, _attr_identifies, _norm_attr
    ground_attribute = TASKS["ground_full"]

    desc = "一名穿白色上衣的行人走在步道上，位于画面中央，旁边是一辆银白色摩托车。"
    boxes = [_Box(0, "行人"), _Box(1, "行人"), _Box(2, "摩托车")]
    q = "描述这辆车的外观、方位和周边。"
    vlm = {0: {"attribute": "穿白色上衣", "color": "白色", "description": desc,
               "describe_q": q, "describe_kind": "full",
               "questions": ["输出穿白色上衣的行人的检测框。"]},
           1: {"attribute": "白色上衣、黑色裤子", "color": "白色", "description": desc,
               "describe_q": q, "describe_kind": "full",
               "questions": ["框出穿白色上衣的行人。"]},
           2: {"attribute": "粉红色", "color": "粉色", "description": desc,
               "describe_q": q, "describe_kind": "full",
               "questions": ["框出粉红色的摩托车。"]}}
    ctx = Ctx(annotation=None, boxes=boxes, grades={}, vlm=vlm,
              all_labels=["行人", "摩托车"], bbox2d=lambda b: [1, 2, 3, 4],
              spatial=lambda b: "中间", rng=random.Random(0),
              measure_words={"行人": "名", "摩托车": "辆"}, min_desc_len=10)

    # 「穿白色上衣」和「白色上衣、黑色裤子」是包含关系 —— 前者同时匹配两个人
    assert not _attr_identifies(ctx, boxes[0], "穿白色上衣")
    assert not _attr_identifies(ctx, boxes[1], "白色上衣、黑色裤子")
    assert _attr_identifies(ctx, boxes[2], "粉红色")

    # 归一化要能看穿「换个说法」：穿白色上衣 == 白色上衣
    assert _norm_attr("穿白色上衣") == _norm_attr("白色上衣")
    assert _norm_attr("正骑行的") == _norm_attr("骑行")

    labels = set()
    for i in range(60):
        ctx.rng = random.Random(i)
        ctx.used = set()
        out = ground_attribute(ctx)
        if out:
            labels.add(out["label"])
    assert labels == {"摩托车"}, f"撞车的行人不该出样本：{labels}"


def test_prompt_forbids_inventing_details():
    """实测出现过「正拉着行李箱向前走」，而图里根本没有行李箱。
    编造比笼统糟糕得多 —— 笼统只是没信息，编造是错误信息，
    而且训练时模型会把它当成事实学下去。"""
    import prompts
    text = prompts.load("vlm_select")
    assert "行李箱" in text, "反面例子要写具体，泛泛地说「不要编造」拦不住"
    assert "不要补全成一个合理的场景" in text
    assert "区分开" in text, "提示词里要要求 attribute 能区分同类目标"




def test_color_check_does_not_judge_dark_or_blown_regions():
    """HSV 的饱和度在明暗两端都不可靠：近黑的像素 (20,10,15) 算出来 S=0.5，
    但它感官上就是黑的。航拍图里阴影、深色车顶一大片落在这里 ——
    实测因此把「白色」「银灰色」这类正确说法误杀，丢弃率虚高到 25%。"""
    from core import colorcheck
    dark_but_saturated = (337, 0.63, 0.20)      # 实测踩到的：几乎全黑
    assert not colorcheck.conflicts("银灰色", dark_but_saturated)
    assert not colorcheck.conflicts("白色", dark_but_saturated)
    blown = (95, 0.5, 0.98)                     # 过曝，色相是噪声
    assert not colorcheck.conflicts("白色", blown)
    # 明度正常时照常判
    normal = (219, 0.85, 0.60)
    assert colorcheck.conflicts("白色", normal)


def test_color_check_uses_chroma_not_saturation_alone():
    """饱和度要和明度一起看。实测踩到 (95°, S=0.47, V=0.39) —— 一块昏暗的偏绿
    区域（小框套在草地边上取到了背景），单看 S=0.47 就判成「明显有颜色」，
    把正确的「白色」误杀了。真正该拦的蓝卡车是 (219°, S=0.85, V=0.78)，
    S×V 一个 0.18 一个 0.66，用彩度一刀就分开了。丢弃率 25% -> 12.9%。"""
    from core import colorcheck
    dim_green = (95, 0.47, 0.39)
    blue_truck = (219, 0.85, 0.78)
    assert dim_green[1] > colorcheck.SAT_ACHROMATIC     # 单看饱和度会误判
    assert not colorcheck.conflicts("白色", dim_green), "昏暗偏绿不该拦白色"
    assert colorcheck.conflicts("白色", blue_truck), "鲜蓝色必须拦下白色"




# ------------------------------------------------- 描述子类型（内容维度）

def test_describe_kinds_load_and_are_registered():
    """七种子类型各一个提示词文件，都要有对应的 ground_* 任务和配比权重。
    加一种就多一种 —— 文件即配置，别的地方忘了登记这项测试会挂。"""
    import yaml
    from core import describe_kinds
    from core.tasks import TASKS, DESCRIBE_KINDS
    kinds = describe_kinds.load_all()
    assert set(kinds) == set(DESCRIBE_KINDS), (sorted(kinds), sorted(DESCRIBE_KINDS))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "config" / "default.yaml").read_text(encoding="utf-8"))
    for k in kinds:
        assert f"ground_{k}" in TASKS, f"ground_{k} 没注册进 TASKS"
        assert f"ground_{k}" in cfg["tasks"], f"ground_{k} 没有配比权重"
        assert kinds[k].answer_spec and kinds[k].q_example and kinds[k].a_example, k


def test_describe_kind_scope_is_enforced():
    """模型很容易把「只说外观」写成「位于画面左侧的一辆红色三轮车」——
    加了方位就又滑回三段式了。没有这道闸，七种跑几轮会退化成同一种。"""
    from core import describe_kinds
    kinds = describe_kinds.load_all()
    ok = "深红色车身，支着一顶白色遮阳篷，车斗敞开着，里面堆着几个纸箱。"
    bad = "位于画面左侧，一辆红色三轮车，旁边有一棵树。"
    assert describe_kinds.answer_violates(kinds["appearance"], ok) == ""
    assert describe_kinds.answer_violates(kinds["appearance"], bad)
    # position 反过来：不该跑去说外观
    assert describe_kinds.answer_violates(kinds["position"], "车身是红色的。")


def test_ground_task_skips_when_model_declines():
    """指派的类型不适合这个目标时模型返回空，这里要跳过换下一个 ——
    不能拿空描述硬凑一条样本。"""
    from core.tasks import TASKS
    ctx = _ctx_with_filtered([_Box(0, "卡车")], {"卡车": 1})
    ctx.vlm[0].update({"attribute": "银灰色", "color": "银灰色",
                       "describe_kind": "part", "describe_q": "", "description": ""})
    assert TASKS["ground_part"](ctx) is None
    ctx.vlm[0].update({"describe_q": "这辆卡车的货斗是什么样的？",
                       "description": "货斗是蓝色金属的，后挡板放了下来，里面空着。"})
    out = TASKS["ground_part"](ctx)
    assert out is not None and out["describe_kind"] == "part"
    # 只有被指派了这个类型的目标才算数
    assert TASKS["ground_state"](ctx) is None


def test_prompt_change_invalidates_the_vlm_cache():
    """缓存的意义是「同一个问题不重复问」，改了提示词就不是同一个问题了。
    少了这一项，改完 prompts/ 重跑会全部命中旧缓存、输出一字未变，
    而且完全没有提示 —— 实测因此白跑了三次。"""
    import importlib, core.vlm_client as m
    target = prompts.PROMPT_DIR / "describe" / "appearance.txt"
    original = target.read_text(encoding="utf-8")
    before = m._prompt_fingerprint()
    try:
        target.write_text(original + "\n# 改一个字\n", encoding="utf-8")
        assert m._prompt_fingerprint() != before, "改提示词没让指纹变"
    finally:
        target.write_text(original, encoding="utf-8")
    assert m._prompt_fingerprint() == before, "改回来指纹要还原"




def test_stats_do_not_reference_deleted_prompts():
    """删提示词时漏掉下游引用，脚本会在用户跑到那一步才炸 ——
    实测 desc_opening.txt 删掉后 dataset_stats.py 直接起不来。
    这项测试扫所有脚本里写死的提示词名，确认它们都还在。"""
    import re as _re
    import prompts
    root = Path(__file__).resolve().parents[1]
    available = set(prompts._index())
    for path in sorted((root / "scripts").glob("*.py")) + \
                sorted((root / "core").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for name in _re.findall(r'prompts\.(?:load|render|load_variants|'
                                r'render_choice|forbidden_of|required_any_of|'
                                r'optional_group_of|max_len_of|path_of)\("(\w+)"',
                                src):
            assert name in available, f"{path.name} 引用了不存在的提示词 {name}.txt"




def test_preview_provenance_reads_results_not_request_count():
    """预览头部标注「哪个模型生成的」。判断依据必须是【拿到了多少条结果】，
    不是【发了多少次请求】—— 全部命中缓存时请求数是 0，但文字确实来自模型，
    说成「没调用任何模型、全部来自模板兜底」是错的。"""
    import json, os, sys, tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from export_samples import _provenance

    d = Path(tempfile.mkdtemp())
    def write(calls):
        (d / "build_report.json").write_text(json.dumps({
            "vlm_calls": calls,
            "vlm_endpoints": {"e": {"model": "Qwen3-VL-8B", "calls": 0}},
        }), encoding="utf-8")
        return "\n".join(_provenance(d / "train.jsonl"))

    # 全部命中缓存：请求 0 次，但结果来自模型
    out = write({"prefetched": 0, "cache": 120, "failed": 0})
    assert "Qwen3-VL-8B" in out and "没有调用任何模型" not in out, out
    # 真的没调用
    out = write({"prefetched": 0, "cache": 0, "failed": 0})
    assert "没有调用任何模型" in out, out
    # 有失败要标出来 —— 这批数据不完整
    out = write({"prefetched": 100, "cache": 0, "failed": 30})
    assert "30 次调用失败" in out, out
    os.unlink(d / "build_report.json"); os.rmdir(d)




def test_no_dead_config_keys():
    """配置文件里不该有没人读的键。

    废弃一个任务时很容易只删代码、忘了删它的配置 —— 留下的键看着像开关，
    改了却毫无反应，比没有这个选项更糟。实测积了 6 组：refer_locate.*、
    sampling.multi_target_*、sampling.negative_ratio、split.group_by_source、
    output.format，全是 refer_locate / SampleBuilder 废弃后留下的。

    【local.yaml.example 也要查】——它才是服务器上被复制的那份。只查
    default.yaml 时，example 里积了四个死键没人发现：temperature_rewrite、
    max_referring_len、neutral_noun、category_hint_words，全是 refer_locate
    和改写步骤废弃后留下的。
    """
    import yaml
    root = Path(__file__).resolve().parents[1]
    src = "\n".join(p.read_text(encoding="utf-8")
                     for p in list((root / "core").rglob("*.py"))
                     + list((root / "scripts").rglob("*.py")))

    # 整段被取走再遍历、键名在代码里拼不出来的段。只有 tasks 是这样：
    # 七个 ground_* 的权重由 f"ground_{k}" 动态生成，源码里找不到字面量。
    # 【这个豁免只能给这一段】—— 早先它是给所有段的（"段名出现过就整段放行"），
    # 于是 quality / vlm 段下的键怎么加都查不出来，实测漏掉了四个死键。
    ENUMERATED_SECTIONS = {"tasks"}

    def dead_keys(cfg):
        dead = []

        def walk(node, prefix=""):
            for k, v in (node or {}).items():
                path = f"{prefix}{k}"
                if isinstance(v, dict) and v:
                    walk(v, path + ".")
                    continue
                # 两种读法：整条点号路径，或叶子名（v.get("api_url") 这种）
                if (f'"{path}"' in src or f"'{path}'" in src
                        or f'"{k}"' in src or f"'{k}'" in src):
                    continue
                if prefix.rstrip(".") in ENUMERATED_SECTIONS:
                    continue
                dead.append(path)

        walk(cfg)
        return dead

    for name in ("default.yaml", "local.yaml.example"):
        cfg = yaml.safe_load((root / "config" / name).read_text(encoding="utf-8"))
        assert not dead_keys(cfg), \
            f"config/{name} 里这些配置项没人读，删掉或者接上：{dead_keys(cfg)}"

    # local.yaml.example 是给用户复制的，不能带着死键
    ex = yaml.safe_load((root / "config" / "local.yaml.example")
                        .read_text(encoding="utf-8")) or {}
    known = set(cfg)
    assert set(ex) <= known, f"example 里有 default 没有的顶层键：{sorted(set(ex) - known)}"




# --------------------------------------------------------------- 流式与进度

def test_streaming_decodes_utf8_not_latin1():
    """流式必须自己按 UTF-8 解码。requests 的 iter_lines(decode_unicode=True)
    用的是 resp.encoding，而服务端 Content-Type 不带 charset 时默认 ISO-8859-1
    —— 中文会变成「ç\x99½è\x89²」这种乱码，不报错、一路写进数据集。实测踩到。"""
    import json as _json
    from config import Config
    from core.vlm_client import VlmClient

    body = _json.dumps({"choices": [{"delta": {"content": "白色车身"}}]},
                       ensure_ascii=False)
    lines = [f"data: {body}".encode("utf-8"), b"", b"data: [DONE]"]

    class _Resp:
        status_code = 200
        encoding = "ISO-8859-1"          # 服务端没给 charset 时 requests 的默认
        def iter_lines(self, decode_unicode=False):
            for b in lines:
                yield b.decode(self.encoding) if decode_unicode else b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Requests:
        @staticmethod
        def post(*a, **k): return _Resp()

    c = VlmClient(Config({"vlm": {"api_url": "http://x/v1/c", "model": "m"}}))
    got = c._post_stream(_Requests, c.endpoints[0], {}, {})
    assert got == "白色车身", f"流式解码坏了：{got!r}"


def test_progress_degrades_outside_tty():
    """重定向到文件 / nohup 时不能刷回车 —— 几万行 \r 会把日志淹掉。
    非 TTY 时按百分比打行。"""
    import io
    from core.progress import Progress
    buf = io.StringIO()                    # StringIO 不是 TTY
    with Progress("测试", 100, stream=buf, min_interval=0) as bar:
        for _ in range(100):
            bar.step()
    out = buf.getvalue()
    assert "\r" not in out, "非 TTY 不该输出回车"
    assert out.count("\n") <= 25, f"打了太多行：{out.count(chr(10))}"
    assert "100/100" in out


def test_progress_is_thread_safe():
    """预取是多线程的，step() 会被并发调用 —— 计数不能丢。"""
    import io
    from concurrent.futures import ThreadPoolExecutor
    from core.progress import Progress
    buf = io.StringIO()
    bar = Progress("并发", 2000, stream=buf, min_interval=0)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: bar.step(), range(2000)))
    bar.close()
    assert bar.done == 2000, bar.done



if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1; print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL  {name}  {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

def test_fine_grained_kinds_are_not_asked_of_hard_targets():
    """`part`/`contrast` 要求看清细部，困难目标上必然写不出来。

    档位限制写在 prompts/describe/<子类型>.txt 的 `#! max-grade:` 里，
    生成阶段据此把不够档的目标从候选里剔掉 —— 否则模型只能编一个部位出来，
    而这种编造从答案文本上看不出来，闸门都拦不住。
    """
    from types import SimpleNamespace

    from core.difficulty import EASY, HARD
    from core.tasks import TASKS

    def build(grade):
        ctx = _ctx_with_filtered([_Box(0, "卡车")], {"卡车": 1})
        ctx.grades[0] = SimpleNamespace(box_index=0, label="卡车", grade=grade)
        ctx.kind_limits = {"part": "medium"}
        ctx.vlm[0].update({"attribute": "银灰色", "color": "银灰色",
                           "describe_kind": "part",
                           "describe_q": "这辆卡车的货斗是什么样的？",
                           "description": "货斗是蓝色金属的，后挡板放了下来。"})
        return ctx

    assert TASKS["ground_part"](build(EASY)) is not None
    assert TASKS["ground_part"](build(HARD)) is None
    # 没配限制的子类型不受影响
    ctx = build(HARD)
    ctx.kind_limits = {}
    assert TASKS["ground_part"](ctx) is not None


def test_part_and_contrast_declare_a_difficulty_ceiling():
    """限制是写在提示词文件里的配置，不是代码里的硬编码 ——
    这一项防的是有人改提示词时把 `#! max-grade:` 顺手删掉。"""
    from core import describe_kinds
    kinds = describe_kinds.load_all()
    limits = describe_kinds.limits(kinds)
    assert limits.get("part") == "medium"
    assert limits.get("contrast") == "medium"
    # 粗粒度的那几种不该有上限，否则困难目标就一条描述都出不来了
    for name in ("appearance", "position", "relation"):
        assert not kinds[name].max_grade


def test_kind_rotation_skips_kinds_this_image_cannot_support():
    """全是困难目标的图上派 `part`，模型只会返回空，白占一个槽位。
    轮转表要跳过去，而不是把槽位浪费掉。"""
    from types import SimpleNamespace

    from core.difficulty import EASY, HARD
    from core.pipeline import _next_kind

    table = [SimpleNamespace(name="appearance", max_grade=""),
             SimpleNamespace(name="part", max_grade="medium"),
             SimpleNamespace(name="position", max_grade="")]

    # 有简单目标：轮转表一个不跳，part 照派
    picked = []
    cursor = 0
    for _ in range(3):
        k, cursor = _next_kind(table, cursor, [EASY, HARD])
        picked.append(k.name)
    assert picked == ["appearance", "part", "position"]

    # 全是困难目标：part 跳过，槽位让给粗粒度的
    picked = []
    cursor = 0
    for _ in range(3):
        k, cursor = _next_kind(table, cursor, [HARD, HARD])
        picked.append(k.name)
    assert "part" not in picked
    assert picked == ["appearance", "position", "appearance"]

    # 一圈都不合适也必须给出一个 —— 让模型自己拒绝，不能不指派
    only_part = [SimpleNamespace(name="part", max_grade="medium")]
    k, cursor = _next_kind(only_part, 0, [HARD])
    assert k.name == "part"


def test_split_is_three_way_and_leaks_nothing_across_groups():
    """val 会被训练框架拿去挑 checkpoint，用它报模型成绩等于自己给自己出卷子。
    test 必须是独立的第三份，而且三份之间同源组一个都不能重叠 ——
    视频抽帧和 Roboflow 增强会让「同一张图」以几十个文件名出现。"""
    from core.pipeline import _split_by_source, _split_stats

    class _Cfg:
        def get_path(self, key, default=None):
            return {"split.val_ratio": 0.2, "split.test_ratio": 0.2}.get(key, default)

    # 12 个来源组，每组 5 条（同一段视频的连续帧）
    samples = [{"images": [f"vid{g:02d}_mp4-{f}.jpg"], "id": f"{g}_{f}"}
               for g in range(12) for f in range(5)]
    train, val, test = _split_by_source(samples, _Cfg(), seed=7)

    assert len(train) + len(val) + len(test) == 60
    assert test and val and train
    stats = _split_stats(train, val, test)
    assert stats["group_overlap"] == 0
    # 每一组要么整组进 train，要么整组进 val/test，不能被切开
    for part in (train, val, test):
        ids = {s["images"][0].split("_mp4-")[0] for s in part}
        others = {s["images"][0].split("_mp4-")[0]
                  for other in (train, val, test) if other is not part
                  for s in other}
        assert not (ids & others)


def test_split_is_reproducible_for_the_same_seed():
    """换个种子重跑就换一批 test，等于每次评估都在换卷子，前后数字没法比。"""
    from core.pipeline import _split_by_source

    class _Cfg:
        def get_path(self, key, default=None):
            return {"split.val_ratio": 0.1, "split.test_ratio": 0.1}.get(key, default)

    samples = [{"images": [f"img{i:03d}.jpg"], "id": str(i)} for i in range(50)]
    a = _split_by_source(samples, _Cfg(), seed=42)
    b = _split_by_source(samples, _Cfg(), seed=42)
    assert [s["id"] for s in a[2]] == [s["id"] for s in b[2]]


def test_sample_metadata_carries_difficulty_and_size_bucket():
    """困难目标只占 10%，报表拆不开就会被 90% 的简单目标稀释成「还行」。
    多框任务取【最小/最难】的那个框 —— 一条样本的难度由它最难的目标决定。"""
    from types import SimpleNamespace

    from core.difficulty import EASY, HARD
    from core.pipeline import _focus_difficulty

    grader = SimpleNamespace(bucket_of=lambda g: "small" if g.equiv_px < 32 else "large")
    gmap = {
        0: SimpleNamespace(grade=EASY, area_ratio=0.04, equiv_px=204.8),
        1: SimpleNamespace(grade=HARD, area_ratio=0.0004, equiv_px=20.5),
    }

    single = _focus_difficulty([0], gmap, grader)
    assert single["difficulty"] == EASY and single["size_bucket"] == "large"

    # 两个框：难度取最难的 hard，尺寸取最小的那个
    multi = _focus_difficulty([0, 1], gmap, grader)
    assert multi["difficulty"] == HARD
    assert multi["size_bucket"] == "small"
    assert multi["area_ratio"] == 0.0004

    # 拒答类没有焦点框，四个字段都得是 None，不能拿 0 冒充
    none = _focus_difficulty([], gmap, grader)
    assert none == {"difficulty": None, "area_ratio": None,
                    "equiv_px": None, "size_bucket": None}


def test_zone_uniqueness_does_not_drive_difficulty():
    """3x3 分区唯一性衡量的是「用模板还是调 VLM 生成指代」，是生成成本，
    不是目标难不难定位。它并进 max() 的后果实测过：部件级标注的数据集
    （机头/机翼/机轮，同类在一张图里天然重复）15.4 万框里 easy 只剩 67 个。"""
    from core.difficulty import EASY, Grader

    class _Cfg:
        def get_path(self, key, default=None):
            return {"difficulty": {}, "quality": {}}.get(key, default)

    class _Box:
        def __init__(self, index, label, cx, cy):
            self.index, self.label, self.cx, self.cy = index, label, cx, cy
            self.area_ratio = 0.02          # 等效 145px，尺寸维度是 easy
        def short_side_px(self, w, h):
            return 100

    # 两个不同类别的大目标挤在同一个 3x3 分区里：分区不唯一，但同类各只有一个
    graded = Grader(_Cfg()).grade_image(
        [_Box(0, "机翼", 0.1, 0.1), _Box(1, "机轮", 0.15, 0.12)], 1000, 1000)
    assert [g.grade for g in graded] == [EASY, EASY]

    # 同类两个挤在同一分区：难度由密集度给出 medium，不因分区不唯一变 hard
    graded = Grader(_Cfg()).grade_image(
        [_Box(0, "机翼", 0.1, 0.1), _Box(1, "机翼", 0.15, 0.12)], 1000, 1000)
    assert all(g.grade == "medium" for g in graded)
    # 但分区信息要留着 —— 它决定这条样本走模板指代还是走 VLM
    assert all(g.unique_in_zone is False for g in graded)
    assert all(g.reasons["referring"] == "needs_vlm" for g in graded)


def test_min_area_ratio_silently_killing_size_reject_is_reported():
    """quality.min_area_ratio 和 difficulty.size_reject_px 都在管「多小算太小」，
    前者先执行。默认的 0.001 换算成等效边长是 32.4px，比 size_reject_px=16 严，
    于是后者完全失效、尺寸维度永远判不出 hard —— 改它一点反应都没有。"""
    from core.difficulty import Grader

    class _Cfg:
        def __init__(self, **q):
            self.q = q
        def get_path(self, key, default=None):
            return {"difficulty": {}, "quality": self.q}.get(key, default)

    conflicts = Grader(_Cfg(min_area_ratio=0.001)).config_conflicts()
    assert any("size_reject_px" in c for c in conflicts)
    assert any("0.000244" in c for c in conflicts)      # 给出该改成多少

    # 对齐之后就不该再报
    assert not Grader(_Cfg(min_area_ratio=0.0002)).config_conflicts()


def test_quota_formula_is_shared_between_build_and_analyze():
    """analyze 预估产出和实际入库必须用同一个公式。早先 analyze 直接拿
    「没被剔除的框数」报数，忽略了困难配额 —— 简单档只有 323 个框时，
    它说「每图 4.16 条」，真实值是 0.02 条，差 170 倍，而且要跑完全量才发现。"""
    from types import SimpleNamespace

    from core.difficulty import EASY, HARD, balance_hard_quota, hard_kept_under_quota

    # 简单档稀少时，配额是全局瓶颈
    assert hard_kept_under_quota(n_simple=323, n_hard=60930, hard_quota=0.10) == 36
    assert hard_kept_under_quota(n_simple=0, n_hard=10000, hard_quota=0.10) == 0
    # 困难档不够时，能留多少留多少
    assert hard_kept_under_quota(n_simple=9000, n_hard=5, hard_quota=0.10) == 5
    assert hard_kept_under_quota(n_simple=100, n_hard=100, hard_quota=0) == 0
    assert hard_kept_under_quota(n_simple=100, n_hard=100, hard_quota=1) == 100

    # balance_hard_quota 必须给出和公式一致的结果
    cand = [SimpleNamespace(g=EASY) for _ in range(323)] \
        + [SimpleNamespace(g=HARD) for _ in range(60930)]
    kept = balance_hard_quota(cand, lambda c: c.g, hard_quota=0.10, seed=1)
    assert sum(1 for c in kept if c.g == HARD) == 36
    assert len(kept) == 359
