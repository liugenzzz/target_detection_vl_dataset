"""ShareGPT 样本的格式契约：图片占位符的位置、角色顺序、语体。

样本的【内容】由 core/tasks.py 的各个任务生成，这里只管【格式】——
落盘前最后一道校验，不合格的整条丢掉并记进构建报告。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import register

IMAGE_TOKEN = "<image>"


def validate_sample(sample: Dict[str, Any],
                    forbid_chat: Sequence[str] = ()) -> List[str]:
    """样本校验。返回问题列表，空列表 = 合格。

    forbid_chat 非空时同时查语体：进训练集的每一句人类问话都必须是指令，
    不能是闲聊。这是第三道也是最后一道语体闸 —— 前两道（扩充问法库时、
    VLM 现场生成问句时）都可能被绕过：有人手改了问法池的 .txt、
    换了提示词、或者直接塞进来一个外部问法库。只有这一道扫的是最终产物。
    """
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
        if forbid_chat and turn.get("from") == "human":
            text = str(turn.get("value", "")).replace(IMAGE_TOKEN, "").strip()
            for bad in register.problems(text, forbid_chat):
                issues.append(f"第 {i} 轮问话不是指令口吻（{bad}）：{text[:24]}")
    return issues
