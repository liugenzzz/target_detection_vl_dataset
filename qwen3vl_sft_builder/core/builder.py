"""三轮 ShareGPT 样本组装。

一条样本 = 一张图 + 一个目标 + 三轮递进对话：
    轮1 识别：用指代锁定目标，问它是什么      -> 类别名
    轮2 定位：承接上一轮，要位置              -> bbox_2d JSON（0~1000）
    轮3 描述：描述该目标的外观与场景          -> VLM 生成的自然语言

为什么轮1 用指代锁定而不是直接问「图中是什么」：一张图里有多个目标时，
直接问「图中是什么」却只答一个，等于在教模型漏报。用指代先锁定对象，
三轮都指向同一个目标，每一轮的答案都是真话。

另有两类变体：
  - 多目标样本（默认 10%）：一次涉及 2~3 个目标，轮2 答案是 JSON 数组。
  - 拒答样本（默认 5%）：问图中不存在的类别，答「图中没有 X」。
    没有这类样本，模型会学到「被问就一定有」的先验，推理时凭空编框。

格式硬约束：<image> 占位符只在第一轮 human 出现一次。多轮里每轮都加会导致
图像 token 重复注入，训练直接崩 —— 这是 ShareGPT 多模态最常见的踩坑点，
build_* 之后由 validate_sample() 强制检查。
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence

import prompts

from .coords import yolo_to_bbox2d
from .referring import template_description, template_referring
from .vlm_client import VlmResult

IMAGE_TOKEN = "<image>"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _turn(role: str, text: str) -> Dict[str, str]:
    return {"from": role, "value": text}


class SampleBuilder:
    def __init__(self, cfg, table, vlm=None):
        self.cfg = cfg
        self.table = table
        self.vlm = vlm
        self.scale = int(cfg.get_path("coords.scale", 1000))
        self.origin = int(cfg.get_path("coords.origin", 0))
        self.image_style = cfg.get_path("output.image_path_style", "filename")
        self.include_meta = bool(cfg.get_path("output.include_metadata", True))

    # -------------------------------------------------------------- 工具
    def _image_value(self, annotation) -> str:
        if self.image_style == "absolute":
            return str(annotation.image_path.resolve())
        if self.image_style == "relative":
            return str(annotation.image_path).replace("\\", "/")
        return annotation.image_path.name

    def _bbox2d(self, box, annotation) -> List[int]:
        return yolo_to_bbox2d(box.cx, box.cy, box.w, box.h,
                              annotation.width, annotation.height,
                              self.scale, self.origin)

    def _base_meta(self, annotation) -> Dict[str, Any]:
        return {
            "source_annotation": annotation.label_path.name,
            "source_image": annotation.image_path.name,
            "image_width": annotation.width,
            "image_height": annotation.height,
            "coordinate_mode": f"qwen_relative_{self.scale}",
            "bbox_scale": self.scale,
            "coord_origin": self.origin,
        }

    def vlm_task(self, annotation, box, grade):
        """构造该目标的 VLM 请求 (image_path, bbox, label, prompt_text)。

        供 pipeline 并发预取用 —— 预取和实际取描述必须走同一套 prompt 渲染，
        否则两边算出的缓存 key 对不上，预取就白做了。
        """
        bbox = self._bbox2d(box, annotation)
        siblings = ""
        if grade.same_label_count > 1:
            siblings = (f"注意：图中还有其他 {grade.same_label_count - 1} 个同为"
                        f"「{box.label}」的目标，指代短语必须能把本目标与它们区分开。\n")
        prompt_text = prompts.render("vlm_describe", bbox=bbox, label=box.label,
                                     siblings_hint=siblings)
        return annotation.image_path, bbox, box.label, prompt_text

    def _resolve_text(self, annotation, box, grade) -> VlmResult:
        """拿到该目标的指代与描述：优先 VLM，失败或未启用时回落模板。"""
        fallback = VlmResult(
            referring=template_referring(box.cx, box.cy, grade.unique_in_zone, box.label),
            description=template_description(box.label, box.cx, box.cy, grade.equiv_px),
            source="template",
        )
        if self.vlm is None:
            return fallback

        image_path, bbox, label, prompt_text = self.vlm_task(annotation, box, grade)
        return self.vlm.describe(image_path, bbox, label, prompt_text, fallback)

    # -------------------------------------------------------------- 单目标
    def build_single(self, annotation, box, grade) -> Dict[str, Any]:
        text = self._resolve_text(annotation, box, grade)
        bbox = self._bbox2d(box, annotation)

        convs = [
            _turn("human", f"{IMAGE_TOKEN}\n"
                           + prompts.render("turn1_identify", referring=text.referring)),
            _turn("gpt", prompts.render("turn1_answer", label=box.label)),
            _turn("human", prompts.render("turn2_locate")),
            _turn("gpt", _json({"bbox_2d": bbox, "label": box.label})),
            _turn("human", prompts.render("turn3_describe", label=box.label)),
            _turn("gpt", text.description),
        ]

        sample: Dict[str, Any] = {
            "id": f"{annotation.stem}_obj{box.index}",
            "images": [self._image_value(annotation)],
            "conversations": convs,
        }
        if self.include_meta:
            sample["metadata"] = dict(
                self._base_meta(annotation),
                sample_type="single",
                box_index=box.index,
                class_id=box.class_id,
                label=box.label,
                difficulty=grade.grade,
                difficulty_reasons=grade.reasons,
                equiv_px=round(grade.equiv_px, 1),
                area_ratio=round(grade.area_ratio, 6),
                same_label_count=grade.same_label_count,
                referring_source="vlm" if text.source in ("vlm", "cache") else "spatial",
                description_source=text.source,
                confusable_class=self.table.is_confusable(box.class_id),
                confusable_group=self.table.confusable_group(box.class_id),
            )
        return sample

    # -------------------------------------------------------------- 多目标
    def build_multi(self, annotation, boxes: Sequence, grades: Sequence
                    ) -> Optional[Dict[str, Any]]:
        """多目标样本。各目标的指代短语必须互不相同 —— 否则「中部左侧那个目标」
        出现两次，模型无法区分问的是哪一个。凑不出互异指代时返回 None，
        这张图退回只出单目标样本。

        VLM 开启时同分区的目标也可能有互异的视觉指代（「白色那艘」/「深色那艘」），
        所以这里校验的是最终指代文本，而不是简单按空间分区排除。
        """
        texts = [self._resolve_text(annotation, b, g) for b, g in zip(boxes, grades)]
        referrings = [t.referring for t in texts]

        # 两道检查，缺一不可：
        # (a) 被选中的目标之间指代互不相同。
        if len(set(referrings)) != len(referrings):
            return None
        # (b) 每个指代在【整张图】范围内也唯一。模板指代是「分区+类别」拼出来的，
        #     两个目标各自在不同分区、但各自分区内都不唯一时，它们的指代互不相同
        #     却各自都能匹配到图中多个目标 —— 只做 (a) 会放过这种样本。
        #     VLM 生成的视觉指代无法在此校验，只能信任模型。
        for text, grade in zip(texts, grades):
            if text.source == "template" and not grade.unique_in_zone:
                return None
        bboxes = [self._bbox2d(b, annotation) for b in boxes]

        referring = "、".join(t.referring for t in texts)
        answer = [{"bbox_2d": bb, "label": b.label} for bb, b in zip(bboxes, boxes)]
        description = "".join(t.description for t in texts)

        convs = [
            _turn("human", f"{IMAGE_TOKEN}\n"
                           + prompts.render("multi_turn1_identify", referring=referring)),
            _turn("gpt", "、".join(b.label for b in boxes) + "。"),
            _turn("human", prompts.render("multi_turn2_locate")),
            _turn("gpt", _json(answer)),
            _turn("human", prompts.render("multi_turn3_describe")),
            _turn("gpt", description),
        ]

        sample: Dict[str, Any] = {
            "id": f"{annotation.stem}_multi{'_'.join(str(b.index) for b in boxes)}",
            "images": [self._image_value(annotation)],
            "conversations": convs,
        }
        if self.include_meta:
            sample["metadata"] = dict(
                self._base_meta(annotation),
                sample_type="multi",
                box_indices=[b.index for b in boxes],
                labels=[b.label for b in boxes],
                target_count=len(boxes),
                difficulty=max((g.grade for g in grades),
                               key=lambda x: ["easy", "medium", "hard"].index(x)),
                description_source=texts[0].source if texts else "template",
            )
        return sample

    # -------------------------------------------------------------- 拒答
    def build_negative(self, annotation, absent_label: str) -> Dict[str, Any]:
        convs = [
            _turn("human", f"{IMAGE_TOKEN}\n"
                           + prompts.render("negative_ask", label=absent_label)),
            _turn("gpt", prompts.render("negative_answer", label=absent_label)),
        ]
        sample: Dict[str, Any] = {
            "id": f"{annotation.stem}_neg_{absent_label}",
            "images": [self._image_value(annotation)],
            "conversations": convs,
        }
        if self.include_meta:
            sample["metadata"] = dict(self._base_meta(annotation),
                                      sample_type="negative", absent_label=absent_label)
        return sample


def validate_sample(sample: Dict[str, Any]) -> List[str]:
    """样本格式校验。返回问题列表，空列表 = 合格。"""
    issues: List[str] = []
    convs = sample.get("conversations") or []

    if not sample.get("images"):
        issues.append("缺少 images 字段")
    if len(convs) < 2 or len(convs) % 2 != 0:
        issues.append(f"对话轮数异常：{len(convs)}")

    image_tokens = sum(t.get("value", "").count(IMAGE_TOKEN) for t in convs)
    if image_tokens != 1:
        issues.append(f"{IMAGE_TOKEN} 出现 {image_tokens} 次，必须恰好 1 次（且在第一轮）")
    elif convs and IMAGE_TOKEN not in convs[0].get("value", ""):
        issues.append(f"{IMAGE_TOKEN} 不在第一轮")

    for i, turn in enumerate(convs):
        expected = "human" if i % 2 == 0 else "gpt"
        if turn.get("from") != expected:
            issues.append(f"第 {i} 轮的 from 应为 {expected}，实为 {turn.get('from')}")
        if not str(turn.get("value", "")).strip():
            issues.append(f"第 {i} 轮内容为空")
    return issues
