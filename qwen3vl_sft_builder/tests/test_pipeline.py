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
