"""全量质检：每一条问答对都让大模型对着原图核对一遍，打分。

为什么必须做：构建阶段的过滤全是【结构性】的 —— 占位符对不对、有没有泄漏
类别名、描述够不够长、框有没有超出画面。这些都不看图。真正只有看图才能发现
的问题，结构过滤一个也拦不住：

    框偏了       坐标合法、格式合法，但框住的是旁边那棵树
    编参照物     「旁边有一辆白色轿车」—— 图里根本没有轿车
    指代不清     图里五辆一模一样的三轮车，问句只说「那辆三轮车」
    类别认错     标注文件写的是卡车，图里其实是公交车

四个维度分别打 1~5 分，低于阈值的挑出来，不直接删 —— 落进 rejected.jsonl
供人工看一眼再决定，因为审核模型自己也会看错。

【按图分组】：一张图上的所有样本合并成一次调用。图片的 base64 是请求里最大
的一块，分开发等于把同一张图传 N 遍。十万条样本按每图 8 条算，
分组后是一万两千多次调用，不分组是十万次。

【别用同一个模型自审】：让写描述的模型来评自己写的描述，它会倾向于认同自己
的输出。配置里 vlm.roles.review 可以给审核单独指定模型或整个模型池。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import prompts

from .sample import IMAGE_TOKEN

logger = logging.getLogger(__name__)

DIMENSIONS = ("correct", "grounded", "clear", "instruction", "needs_image")

# 综合分只看这几个维度的最小值。needs_image 不进综合分 —— 它衡量的是
# 「这条样本有没有训练价值」，不是「这条样本对不对」。一条不看图也能答对的
# 拒答样本并没有错，只是没用；该不该留由 review.min_dimension 单独卡。


def render_samples(group: List[Dict[str, Any]]) -> str:
    """把一张图上的若干条样本拼成给审核模型看的文本。"""
    blocks = []
    for i, s in enumerate(group):
        turns = []
        for t in s.get("conversations", []):
            who = "问" if t.get("from") == "human" else "答"
            turns.append(f"  {who}：{str(t.get('value', '')).replace(IMAGE_TOKEN, '').strip()}")
        blocks.append(prompts.render(
            "review_sample", id=i,
            task=(s.get("metadata") or {}).get("task_type", "?"),
            turns="\n".join(turns)))
    return "\n\n".join(blocks)


def parse(raw: str, n: int) -> Optional[Dict[int, Dict[str, Any]]]:
    """解析审核返回。拿不到就返回 None —— 宁可这一组不判，也不能把
    解析失败当成满分放行。"""
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    reviews = data.get("reviews") if isinstance(data, dict) else None
    if not isinstance(reviews, list):
        return None

    out: Dict[int, Dict[str, Any]] = {}
    for item in reviews:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < n:
            continue          # 模型编了个不存在的编号
        scores = {}
        for dim in DIMENSIONS:
            try:
                v = int(item.get(dim))
            except (TypeError, ValueError):
                continue
            scores[dim] = max(1, min(5, v))
        if not scores:
            continue
        scores["issue"] = str(item.get("issue") or "").strip()
        core_dims = [d for d in DIMENSIONS if d != "needs_image" and d in scores]
        scores["score"] = min(scores[d] for d in core_dims) if core_dims else 3
        out[idx] = scores
    return out or None


def verdict(scores: Dict[str, Any], min_score: int,
            min_dim: Dict[str, int]) -> Tuple[bool, str]:
    """判定一条样本过不过。返回 (是否通过, 原因)。

    综合分取【各维度的最小值】而不是平均：一条描述编造了参照物（grounded=1）
    但问句写得漂亮（instruction=5），平均下来还有 3 分多，照样进训练集。
    质检要看短板。
    """
    for dim, floor in min_dim.items():
        got = scores.get(dim)
        if got is not None and got < floor:
            return False, f"{dim}={got} 低于下限 {floor}"
    if scores.get("score", 5) < min_score:
        return False, f"综合分 {scores['score']} 低于 {min_score}"
    return True, ""


def summarize(reviewed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总质检结果，进 review_report.json。"""
    scored = [s for s in reviewed if (s.get("review") or {}).get("score")]
    if not scored:
        return {"scored": 0}

    dim_avg = {}
    for dim in DIMENSIONS:
        vals = [s["review"][dim] for s in scored if dim in s["review"]]
        if vals:
            dim_avg[dim] = round(sum(vals) / len(vals), 2)

    passed = [s for s in scored if s["review"].get("passed")]
    by_task: Dict[str, List[int]] = defaultdict(list)
    for s in scored:
        by_task[(s.get("metadata") or {}).get("task_type", "?")].append(
            s["review"]["score"])

    dist = Counter(s["review"]["score"] for s in scored)
    reasons = Counter(s["review"].get("reason", "") for s in scored
                      if not s["review"].get("passed"))
    return {
        "scored": len(scored),
        "unscored": len(reviewed) - len(scored),
        "passed": len(passed),
        "rejected": len(scored) - len(passed),
        "pass_rate": round(len(passed) / len(scored), 4),
        "score_avg": round(sum(s["review"]["score"] for s in scored) / len(scored), 2),
        "score_dist": {str(k): dist[k] for k in sorted(dist)},
        "dimension_avg": dim_avg,
        "by_task": {t: {"n": len(v), "avg": round(sum(v) / len(v), 2),
                        "pass_rate": round(
                            sum(1 for s in scored
                                if (s.get("metadata") or {}).get("task_type") == t
                                and s["review"].get("passed")) / len(v), 4)}
                    for t, v in sorted(by_task.items())},
        "top_reject_reasons": dict(reasons.most_common(8)),
    }


def group_by_image(samples: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按图片分组。图片的 base64 是请求里最大的一块，同一张图的样本合并成
    一次调用，十万条样本的审核开销从十万次降到一万多次。"""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        images = s.get("images") or []
        groups[images[0] if images else ""].append(s)
    return dict(groups)


def resolve_image(value: str, images_dir: Optional[Path]) -> Optional[Path]:
    """把样本里的 images 字段还原成磁盘上的真实路径。

    output.image_path_style 默认是 filename，落盘的是裸文件名（LLaMA-Factory
    在训练时按 media_dir 拼），质检要读图就得自己拼回去。
    """
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    if images_dir:
        cand = images_dir / p.name
        if cand.exists():
            return cand
    return p if p.exists() else None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    # newline="\n"：见 pipeline._write_jsonl 的说明
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
