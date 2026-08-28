"""回归测试。不依赖外部数据和 VLM 服务，可以在服务器上直接跑：

    python -m pytest tests/ -q     或     python tests/test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.builder import IMAGE_TOKEN, validate_sample
from core.classes import _detect_confusable, _one_char_apart
from core.coords import yolo_to_bbox2d, zone_of
from core.grouping import source_group_key
from core.referring import spatial_phrase
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
    groups = _detect_confusable({3: "一般人员", 9: "人员", 18: "军事人员",
                                 23: "切管器", 24: "切管机",
                                 5: "两栖战舰", 6: "主力战舰", 2: "suv"})
    assert 9 in groups and "军事人员" in groups[9]      # 包含关系
    assert 23 in groups and "切管机" in groups[23]      # 一字之差
    assert 5 not in groups                              # 差两字，视觉可区分
    assert _one_char_apart("压接钳", "压管钳")
    assert not _one_char_apart("剪刀", "剪线钳")


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
    from core.vlm_client import VlmResult
    c = _client(cache_dir="")            # 故意不配磁盘缓存
    assert c._cache_path(Path("a.jpg"), [1, 2, 3, 4]) is None

    key = c._key(Path("a.jpg"), [1, 2, 3, 4])
    c._memory[key] = VlmResult("视觉指代", "视觉描述", "vlm")
    c._prefetch_done = True

    fallback = VlmResult("模板指代", "模板描述", "template")
    got = c.describe(Path("a.jpg"), [1, 2, 3, 4], "船", "prompt", fallback)
    assert got.referring == "视觉指代", "内存里的预取结果必须被取用"
    assert got.description == "视觉描述"


def test_describe_never_requests_after_prefetch():
    """审查发现的次生问题：预取失败的目标若在串行组装阶段重新发请求，
    10 万条里 1% 失败就是 1000 次带重试的串行调用，能把整批任务拖垮。"""
    from core.vlm_client import VlmResult
    c = _client(cache_dir="")
    c._prefetch_done = True
    c._request = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("组装阶段不允许发起网络请求"))
    fallback = VlmResult("模板指代", "模板描述", "template")
    assert c.describe(Path("b.jpg"), [1, 2, 3, 4], "船", "p", fallback).source == "template"


def test_empty_json_is_not_success():
    """审查发现 #5：形状不对的 JSON 会解析成两个空串。若当成功，
    会写入空缓存，之后每次运行都命中它，把该目标永久钉死在模板文本上。"""
    from core.vlm_client import _parse_vlm_json
    assert _parse_vlm_json('{"referring":"","description":""}') is None
    assert _parse_vlm_json('{"ref":"x","desc":"y"}') is None          # 键名写错
    ok = _parse_vlm_json('{"referring":"","description":"只有描述"}')  # 部分有值仍算成功
    assert ok is not None and ok.description == "只有描述"


def test_multi_rejects_image_wide_ambiguous_referring():
    """审查发现 #3：两个目标各在不同分区、但各自分区内都不唯一时，
    它们的指代互不相同、却各自都匹配图中多个目标。只比较被选中的目标不够。"""
    from core.builder import SampleBuilder
    from config import load_config

    class FakeGrade:
        def __init__(self, uniq):
            self.unique_in_zone = uniq
            self.same_label_count = 3
            self.equiv_px = 80.0
            self.grade = "medium"
            self.reasons = {}
            self.box_index = 0

    class FakeBox:
        def __init__(self, i, cx, cy):
            self.index, self.cx, self.cy = i, cx, cy
            self.w = self.h = 0.1
            self.label, self.class_id = "船", 1

    class FakeAnn:
        stem = "t"
        width = height = 640
        image_path = Path("t.jpg")
        label_path = Path("t.txt")

    class FakeTable:
        is_confusable = staticmethod(lambda cid: False)
        confusable_group = staticmethod(lambda cid: [])

    b = SampleBuilder(load_config(), FakeTable(), None)   # vlm=None -> 走模板
    boxes = [FakeBox(0, 0.2, 0.2), FakeBox(1, 0.8, 0.8)]  # 不同分区
    assert b.build_multi(FakeAnn(), boxes, [FakeGrade(False), FakeGrade(False)]) is None, \
        "分区内不唯一的模板指代必须被拒绝"
    assert b.build_multi(FakeAnn(), boxes, [FakeGrade(True), FakeGrade(True)]) is not None, \
        "分区内唯一时应正常生成"



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



def test_referring_must_not_leak_the_label():
    """实战发现的严重缺陷：VLM 生成的指代短语里带了类别名，例如
        问：图中靠近树木的那辆【三轮车】是什么？   答：是tricycle。
    等于把答案写进了问题里，模型学不到识别能力，只学会把中文类别名翻译成英文。
    提示词里已明令禁止，但提示词管不住模型，代码这一层必须兜底。"""
    from core.referring import leaks_label

    # 类别名整体出现
    assert leaks_label("靠近树木的那辆三轮车", "三轮车")
    assert leaks_label("白色轿车左侧的银色面包车", "面包车")
    # 两字以上尾缀出现
    assert leaks_label("穿红色上衣的那个人员", "军事人员")
    # 单字尾缀出现在结尾
    assert leaks_label("画面中部那艘船", "其它辅助船")
    assert leaks_label("岸边那艘战舰", "主力战舰")


def test_referring_leak_check_avoids_false_positives():
    """单字尾缀只认结尾，否则「车」「船」这类高频字会大量误判 ——
    「车头朝左」并没有泄漏「卡车」，这类合格指代不能被误杀。"""
    from core.referring import leaks_label

    assert not leaks_label("停在斜坡上、车头朝左的那个目标", "卡车")
    assert not leaks_label("停在白色轿车左侧、车身银色的那个目标", "其它辅助船")
    assert not leaks_label("广场上被一个人骑着的那个目标", "自行车")
    assert not leaks_label("水面上行驶、拖着白色尾迹的那个目标", "两栖战舰")
    # 英文类别名无法与中文指代比对，不做判断（返回 False 而不是误杀）
    assert not leaks_label("parked next to the white car", "van")
    # 空值不该崩
    assert not leaks_label("", "卡车") and not leaks_label("那个目标", "")


def test_prompt_forbids_naming_the_category():
    """提示词必须明确禁止在指代里出现类别名，并给出正反例 ——
    只写「优先用外观特征」是不够的，模型会照样用类别名。"""
    import prompts

    text = prompts.load("vlm_describe")
    assert "绝对不能出现" in text, "必须是硬禁止，不能只是「优先」"
    assert "反例" in text and "正例" in text, "要给出正反例，光讲规则模型抓不住"



def test_overlong_referring_is_rejected():
    """指代的职责只是唯一锁定目标，能区分就够了。模型倾向把看到的特征全堆上去，
    实测会写出「画面中上部，停在白色轿车和黑色轿车之间，靠近一棵绿树的那个目标」
    这种三从句 30 字指代 —— 它是问题的主语，堆满定语让问题读不下去，
    多目标样本三个拼起来接近 80 字。"""
    from core.referring import too_long

    assert too_long("画面中上部，停在白色轿车和黑色轿车之间，靠近一棵绿树的那个目标")
    assert too_long("画面右下角、靠近充气水池边缘、带有浅色顶棚的那个目标")
    # 一到两个特征的合格指代不能被误杀
    assert not too_long("车身银色的那个目标")
    assert not too_long("顶部有绿色遮篷的那个目标")
    assert not too_long("停在白色轿车后方、车身银色的那个目标")
    assert not too_long("") and not too_long(None)


def test_referring_limit_is_configurable():
    """上限要能在 config 里调，不能写死在代码里。"""
    from config import load_config
    from core.referring import too_long

    limit = load_config().get_path("quality.max_referring_len")
    assert isinstance(limit, int) and limit > 0, "config 里必须有这一项"
    assert too_long("一" * (limit + 1), limit)
    assert not too_long("一" * limit, limit)


def test_prompt_demands_short_referring():
    """提示词必须要求指代尽量短并给出「太长」的反例 ——
    只给字数上限不够，模型会正好写满上限。"""
    import prompts

    text = prompts.load("vlm_describe")
    assert "越短越好" in text
    assert "反例二" in text, "要给出「太长」的反例，光说字数模型抓不住"



def test_referring_rejects_category_hint_words():
    """类别名泄漏的隐蔽形式：字面上没写类别名，但部件名和动作词把答案暴露了。

        「车身银色的那个」          「车身」把答案限定成车辆
        「一名穿粉衣的人正骑行的那个」 「骑行」限定成自行车/摩托车/三轮车

    指代只该用位置、颜色、与周围物体的相对关系，这三类都不暗示类别。"""
    from config import load_config
    from core.referring import implies_category

    words = load_config().get_path("quality.category_hint_words")
    assert words, "config 里必须有 category_hint_words"

    assert implies_category("车身银色的那个", words) == "车身"
    assert implies_category("一名穿粉色上衣的人正骑行的那个", words) == "骑行"
    assert implies_category("停在大石头后方的那个", words) == "停在"
    # 地标式指代必须放行 —— 用别的东西定位目标，没有暴露目标自己是什么
    assert implies_category("白色轿车旁边的那个", words) is None
    assert implies_category("戴红色帽子的人左边的那个", words) is None
    assert implies_category("画面右下角、银色的那个", words) is None


def test_template_referring_never_leaks_the_label():
    """曾经的严重 bug：模板指代在分区内不唯一时拼「那个{label}」来消歧，
    生成出「图中上方右侧那个van是什么？」—— 答案就在问题里。
    更糟的是 leaks_label() 检测到 VLM 指代泄漏后回落的正是这个模板，原地打转。"""
    from core.referring import template_referring

    for uniq in (True, False):
        r = template_referring(0.8, 0.2, uniq)
        assert "van" not in r and "面包车" not in r, f"模板指代不能带类别名：{r}"
    assert template_referring(0.8, 0.2) == "上方右侧那个"


def test_neutral_noun_is_not_annotation_jargon():
    """「目标」是标注术语，正常人不会问「那个目标是什么」。
    默认用光杆的「那个」——「画面右下角、银色的那个是什么？」。
    不用「物体」是因为类别里有人员，把人叫「那个物体」很别扭。"""
    from config import load_config
    from core.referring import template_referring

    noun = load_config().get_path("quality.neutral_noun")
    assert noun and "目标" not in noun
    assert "目标" not in template_referring(0.5, 0.5)


def test_ambiguous_single_sample_is_dropped():
    """分区内不唯一、VLM 又没给出可用视觉指代时，模板指代会同时匹配多个目标，
    问题有多个正确答案，是坏数据。此前靠拼类别名消歧，那比歧义更糟。
    现在整条丢弃。"""
    from pathlib import Path
    from config import load_config
    from core.builder import SampleBuilder

    class G:
        def __init__(self, uniq): self.unique_in_zone = uniq
        same_label_count = 3; equiv_px = 80.0; area_ratio = 0.01
        grade = "medium"; reasons = {}; box_index = 0

    class B:
        index = 0; cx = cy = 0.5; w = h = 0.1; label = "面包车"; class_id = 4

    class A:
        stem = "t"; width = height = 640
        image_path = Path("t.jpg"); label_path = Path("t.txt")

    class T:
        is_confusable = staticmethod(lambda c: False)
        confusable_group = staticmethod(lambda c: [])

    b = SampleBuilder(load_config(), T(), None)      # vlm=None -> 全走模板
    assert b.build_single(A(), B(), G(True)) is not None, "分区内唯一应正常生成"
    assert b.build_single(A(), B(), G(False)) is None, "分区内不唯一应丢弃"
    assert b.ambiguous_dropped == 1







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
        optional = prompts.has_flag(name, "optional-refer")
        req_any = prompts.required_any_of(name)
        seeds = prompts.load_variants(name)
        assert forbidden, f"{name} 没写 #! forbid，语义闸是空的"
        for v in seeds:
            assert phrase_bank.accept(name, v, required, 30, [], forbidden,
                                      optional, req_any), \
                f"{name} 自己的种子 {v!r} 过不了自己的规则"




def test_bank_optional_refer_is_all_or_nothing():
    req = ["label", "mw"]
    # 承接上文，整句不带占位符 —— 允许
    assert phrase_bank.accept("x", "它周围是什么情况？", req, 30, [], (), True)
    # 只带一半：剩下「这{mw}」或量词丢了的「这三轮车」，比不提还糟
    assert not phrase_bank.accept("x", "说说这{mw}。", req, 30, [], (), True)
    assert not phrase_bank.accept("x", "说说这{label}。", req, 30, [], (), True)
    # 没声明 optional-refer 的池子，一个都不能少
    assert not phrase_bank.accept("x", "它在哪？", req, 30, [], (), False)


def test_bank_install_drops_bad_lines(tmp_path=None):
    """问法库是给人手改的，改坏了不能带着跑 —— install 会再校验一遍。"""
    import tempfile, os, prompts
    from config import Config
    banks = {"ask_describe": [
        "具体说说这{mw}{label}。",          # 好
        "那{mw}{label}在哪？",              # 禁用词：跑到「问位置」上去了
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
                              (), False, req_any)
    assert not phrase_bank.accept("x", "框出图中的{label}。", ["label"], 30, [],
                                  (), False, req_any)




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