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