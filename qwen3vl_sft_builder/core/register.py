"""语体闸：问答对必须是【指令】，不是聊天。

数据集里的每一句人类问话，都是下达给视觉模型的指令，参照 RefCOCO / Qwen
grounding 训练数据的口吻 —— 祈使句（「框出图中的卡车。」）或直接的疑问句
（「卡车在图中的什么位置？」）。「诶那辆卡车在哪儿？」「帮我把人员框出来」
这种闲聊腔一旦进了训练集，模型学到的问句分布就跟推理时对不上。

这道闸有三个入口，共用同一份禁用词（config 的 phrase_banks.forbid_global）：

    1. 扩充问法库时       —— 生成结果逐条过滤
    2. VLM 现场生成问句时 —— ground_attribute 的问句是按图生成的，不走问法库
    3. 样本落盘前         —— 兜底，扫每一条人类问话

第 3 道是兜底：前两道都可能被绕过（手改问法池的 .txt、换了提示词），
只有落盘前这一道能保证「进训练集的每一句都是指令」。
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# 句末必须有标点。模型的元话语（「以下是几种写法：」）常常没有。
_TERMINATED = re.compile(r"[。？！?!]$")
# 模型在讲「我要写几种说法」，而不是在写那句说法本身
META_WORDS: Tuple[str, ...] = ("以下", "如下", "例如", "比如", "示例", "写法",
                               "说法", "第一句", "这条", "这句", "上面", "下面", "注意")


def chat_hits(text: str, forbidden: Sequence[str]) -> List[str]:
    """返回这句话里命中的闲聊词。空列表 = 合格。"""
    return [w for w in forbidden if w in text]


def problems(text: str, forbidden: Sequence[str] = ()) -> List[str]:
    """一句人类问话的全部语体问题。空列表 = 合格。"""
    out = []
    hits = chat_hits(text, forbidden)
    if hits:
        out.append("闲聊词 " + "/".join(hits))
    if any(w in text for w in META_WORDS):
        out.append("元话语")
    if not _TERMINATED.search(text.strip()):
        out.append("句末缺标点")
    return out


def is_instruction(text: str, forbidden: Sequence[str] = ()) -> bool:
    return not problems(text, forbidden)
