"""数据集体检：不调模型、纯离线算的客观指标。

scripts/review.py 是【主观】质检 —— 让大模型看图打分。这个模块是【客观】体检，
对标公开数据集通用的那几个指标。两者互补，缺一不可：

  主观质检能发现「框住的是旁边那棵树」，客观体检发现不了；
  客观体检能发现「347 个类别只覆盖了 40 个」「85% 的框挤在画面中间」
  「描述里提到了标注文件里根本没有的类别」，主观质检逐条打分时看不出来 ——
  那些是【分布层面】的问题，单看任何一条样本都是好的。

指标来源：

  CHAIR (Rohrbach et al. 2018)  描述里提到的物体有多少不在图中。
                                原版比对 COCO 标注，我们比对 YOLO 标注文件，
                                同一个思路且更严格 —— 标注文件就是真值。
  Distinct-n (Li et al. 2016)   词汇多样性。不同 n-gram 数 / 总 n-gram 数。
  POPE (Li et al. 2023)         存在性问答的正负比与负采样难度。
  长尾覆盖                       类别覆盖率与基尼系数。

全部零成本，跑一次几秒钟。
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_BOX = re.compile(r'"bbox_2d"\s*:\s*\[([0-9,\s.-]+)\]')


def _turns(sample, who) -> List[str]:
    return [str(t.get("value", "")) for t in sample.get("conversations", [])
            if t.get("from") == who]


def chair(samples: Sequence[Dict[str, Any]],
          truth: Dict[str, set], all_labels: Sequence[str]) -> Dict[str, Any]:
    """CHAIR 式幻觉率：模型答案里提到的类别，有多少不在该图的标注里。

    truth: {图片名: {该图标注里出现过的类别名}}

    只统计【类别表里的词】—— 描述里的「斑马线」「路灯杆」不在类别表中，
    无从判定真假，不算幻觉也不算正确。这会低估真实幻觉率，但绝不会误报，
    宁可漏报也不能让一个客观指标出现假阳性。
    """
    vocab = sorted((l for l in all_labels if len(l) >= 2), key=len, reverse=True)
    hit = miss = 0
    sent_bad = 0
    worst: Counter = Counter()
    for s in samples:
        image = (s.get("images") or [""])[0]
        gt = truth.get(image)
        if gt is None:
            continue
        mentioned = set()
        for text in _turns(s, "gpt"):
            if text.strip().startswith(("{", "[")):
                continue                       # 框答案里的 label 来自标注，不是生成的
            for label in vocab:
                if label in text:
                    mentioned.add(label)
        bad = mentioned - gt
        hit += len(mentioned & gt)
        miss += len(bad)
        if bad:
            sent_bad += 1
            worst.update(bad)
    total = hit + miss
    n = sum(1 for s in samples if (s.get("images") or [""])[0] in truth)
    return {
        # CHAIR_i：提到的类别中说错的比例
        "chair_i": round(miss / total, 4) if total else 0.0,
        # CHAIR_s：有几条样本至少说错一个
        "chair_s": round(sent_bad / n, 4) if n else 0.0,
        "mentions_total": total, "mentions_wrong": miss,
        "top_hallucinated": dict(worst.most_common(8)),
    }


# Distinct-n 在【固定条数】的子样本上算。原始 Distinct-n 有语料规模偏差：
# 语料越大，分母涨得比分子快，1 千条和 10 万条算出来必然是后者低，
# 那是规模造成的，不是多样性真的下降。固定子样本量之后跨版本才可比。
DISTINCT_SAMPLE = 2000


def distinct_n(texts: Sequence[str], n: int = 2, seed: int = 0) -> float:
    """Distinct-n 词汇多样性：不同 n-gram 数 / 总 n-gram 数。

    中文按字切 —— 分词器的选择会左右这个数，按字切没有这个自由度，跨版本可比。
    数值本身不跟英文数据集横向比，看的是我们自己的变化趋势。
    """
    if len(texts) > DISTINCT_SAMPLE:
        texts = random.Random(seed).sample(list(texts), DISTINCT_SAMPLE)
    grams: Counter = Counter()
    for t in texts:
        chars = re.sub(r"\s+", "", t)
        for i in range(len(chars) - n + 1):
            grams[chars[i:i + n]] += 1
    total = sum(grams.values())
    return round(len(grams) / total, 4) if total else 0.0


def coverage(samples: Sequence[Dict[str, Any]], all_labels: Sequence[str]) -> Dict[str, Any]:
    """类别覆盖与长尾。347 个类别只练到 40 个，训出来的模型就只认那 40 个。"""
    used = Counter(s.get("metadata", {}).get("label") for s in samples
                   if s.get("metadata", {}).get("label"))
    counts = sorted(used.values())
    n = len(counts)
    if n:
        cum = sum((i + 1) * c for i, c in enumerate(counts))
        gini = round((2 * cum) / (n * sum(counts)) - (n + 1) / n, 4)
    else:
        gini = 0.0
    return {
        "classes_in_table": len(all_labels),
        "classes_covered": n,
        "coverage_rate": round(n / len(all_labels), 4) if all_labels else 0.0,
        # 基尼系数：0 = 每个类别样本数一样多，1 = 全挤在一个类别上
        "gini": gini,
        "top5": dict(used.most_common(5)),
        "tail_lt_10": sum(1 for c in counts if c < 10),
        "never_used": [l for l in all_labels if l not in used][:20],
    }


def box_distribution(samples: Sequence[Dict[str, Any]], scale: int = 1000) -> Dict[str, Any]:
    """框的空间与尺寸分布。85% 的框挤在画面中央的话，模型学到的是「往中间猜」——
    单看每一条样本都没问题，只有汇总才看得出来。"""
    zones: Counter = Counter()
    areas: List[float] = []
    for s in samples:
        for text in _turns(s, "gpt"):
            for m in _BOX.finditer(text):
                try:
                    x1, y1, x2, y2 = [float(v) for v in m.group(1).split(",")]
                except ValueError:
                    continue
                cx, cy = (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale
                zones[f"{'上中下'[min(2, int(cy * 3))]}{'左中右'[min(2, int(cx * 3))]}"] += 1
                areas.append((x2 - x1) * (y2 - y1) / (scale * scale))
    total = sum(zones.values())
    areas.sort()
    def pct(p):
        return round(areas[int(len(areas) * p)], 5) if areas else 0.0
    return {
        "boxes_total": total,
        "zone_share": {k: round(v / total, 4) for k, v in zones.most_common()} if total else {},
        # 九宫格里最挤的那一格占了多少。均匀是 1/9≈0.11，超过 0.25 就该看看数据源
        "max_zone_share": round(max(zones.values()) / total, 4) if total else 0.0,
        "area_p10": pct(0.10), "area_p50": pct(0.50), "area_p90": pct(0.90),
    }


def answer_shape(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """答案侧的形态分布。问句多样性我们一直在盯，答案侧同样会退化 ——
    描述全是「位于画面左下角，一辆…」一个句式，模型学到的就是那个句式。"""
    texts, lens = [], []
    starts: Counter = Counter()
    for s in samples:
        for t in _turns(s, "gpt"):
            if t.strip().startswith(("{", "[")):
                continue
            texts.append(t)
            lens.append(len(t))
            starts[t[:4]] += 1
    lens.sort()
    return {
        "text_answers": len(texts),
        "distinct_rate": round(len(set(texts)) / len(texts), 4) if texts else 0.0,
        # 在固定 2000 条子样本上算，跨版本可比
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "distinct_sample_size": min(len(texts), DISTINCT_SAMPLE),
        "len_p10": lens[len(lens) // 10] if lens else 0,
        "len_p50": lens[len(lens) // 2] if lens else 0,
        "len_p90": lens[len(lens) * 9 // 10] if lens else 0,
        # 开头四个字最集中的几种。一种开头占比过高说明句式在退化
        "top_openings": {k: round(v / len(texts), 4)
                         for k, v in starts.most_common(5)} if texts else {},
    }


def pope_balance(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """存在性问答的正负平衡与负采样难度（POPE 的思路）。

    正负失衡时模型会学成「一律答有」或「一律答没有」；负样本全是不相干的类别
    （航拍街景问「有没有N95防护口罩」）时，答「没有」不需要看图，学不到东西。
    """
    ex = [s for s in samples
          if s.get("metadata", {}).get("task_type") == "exist_negative"]
    if not ex:
        return {}
    pos = sum(1 for s in ex if s["metadata"].get("polarity") == "positive")
    neg = len(ex) - pos
    hard = sum(1 for s in ex if s["metadata"].get("hard_negative"))
    return {"total": len(ex), "positive": pos, "negative": neg,
            "positive_rate": round(pos / len(ex), 4),
            "hard_negative_rate": round(hard / neg, 4) if neg else 0.0}


def report(samples: Sequence[Dict[str, Any]], truth: Dict[str, set],
           all_labels: Sequence[str], scale: int = 1000) -> Dict[str, Any]:
    return {
        "samples": len(samples),
        "hallucination_chair": chair(samples, truth, all_labels),
        "class_coverage": coverage(samples, all_labels),
        "box_distribution": box_distribution(samples, scale),
        "answer_shape": answer_shape(samples),
        "exist_balance": pope_balance(samples),
    }
