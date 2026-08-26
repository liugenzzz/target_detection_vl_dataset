"""把各策略的输出跟数据集自带的标准答案(ground truth)比对，算出准确率。

解决的问题：肉眼看验证图只能判断"框得像不像"，判断不了"该框的漏了多少"。
漏检在预标注场景里代价更大 —— 误检删一下就行，漏检要人工重新画框。

用法：

    python eval.py --gt ./coco128/labels/train2017 --results ./results

    # 只评估某个策略
    python eval.py --gt ./coco128/labels/train2017 --results ./results --strategy direct

    # 调整判定为"框对了"的 IoU 阈值（默认 0.5）
    python eval.py --gt ... --results ... --iou 0.5

输出：
  - 控制台打印各策略的召回率/精确率/F1
  - eval_summary.csv    各策略汇总对比
  - eval_per_class.csv  按类别拆开，看哪些类别模型认不准

依赖：只用标准库，不用装东西。
"""

import argparse
import csv
import os
from collections import defaultdict


def load_yolo_txt(path):
    """读 YOLO 格式标注：class_id cx cy w h（都是归一化值）。"""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            boxes.append({"class_id": cid, "xyxy": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)})
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thr):
    """贪心匹配预测框和真值框。

    返回 (类别+位置都对的数量, 位置对但类别错的数量,
          匹配上的真值索引集合, 匹配上的预测索引集合)
    """
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            v = iou(p["xyxy"], g["xyxy"])
            if v >= iou_thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)

    used_p, used_g = set(), set()
    tp = 0
    wrong_class = 0
    class_pairs = []

    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        if preds[pi]["class_id"] == gts[gi]["class_id"]:
            tp += 1
        else:
            wrong_class += 1
            class_pairs.append((gts[gi]["class_id"], preds[pi]["class_id"]))

    return tp, wrong_class, used_g, used_p, class_pairs


def evaluate(gt_dir, labels_dir, iou_thr):
    stats = {
        "images": 0, "gt_boxes": 0, "pred_boxes": 0,
        "tp": 0, "wrong_class": 0, "missed": 0, "false_pos": 0,
    }
    per_class = defaultdict(lambda: {"gt": 0, "tp": 0, "missed": 0, "fp": 0})
    confusions = defaultdict(int)

    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".txt"))
    for name in gt_files:
        pred_path = os.path.join(labels_dir, name)
        if not os.path.exists(pred_path):
            continue

        gts = load_yolo_txt(os.path.join(gt_dir, name))
        preds = load_yolo_txt(pred_path)

        stats["images"] += 1
        stats["gt_boxes"] += len(gts)
        stats["pred_boxes"] += len(preds)

        for g in gts:
            per_class[g["class_id"]]["gt"] += 1

        tp, wc, used_g, used_p, cpairs = match(preds, gts, iou_thr)
        stats["tp"] += tp
        stats["wrong_class"] += wc
        stats["missed"] += len(gts) - len(used_g)
        stats["false_pos"] += len(preds) - len(used_p)

        for gi, g in enumerate(gts):
            if gi not in used_g:
                per_class[g["class_id"]]["missed"] += 1
        for pi, p in enumerate(preds):
            if pi not in used_p:
                per_class[p["class_id"]]["fp"] += 1
        for gcid, pcid in cpairs:
            confusions[(gcid, pcid)] += 1

    # per-class tp：该类真值数减去漏检数
    for cid, d in per_class.items():
        d["tp"] = max(0, d["gt"] - d["missed"])

    recall = stats["tp"] / stats["gt_boxes"] if stats["gt_boxes"] else 0.0
    precision = stats["tp"] / stats["pred_boxes"] if stats["pred_boxes"] else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    stats["recall"] = round(recall, 4)
    stats["precision"] = round(precision, 4)
    stats["f1"] = round(f1, 4)

    return stats, per_class, confusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="标准答案的 labels 目录")
    ap.add_argument("--results", default="./results", help="batch_run.py 的输出目录")
    ap.add_argument("--only", default=None, help="多组结果时只评估某一组")
    ap.add_argument("--iou", type=float, default=0.5, help="判定框对了的 IoU 阈值")
    args = ap.parse_args()

    if not os.path.isdir(args.gt):
        print(f"找不到标准答案目录：{args.gt}")
        return

    # 支持两种目录结构：
    #   results/labels/           单策略（当前默认）
    #   results/<名字>/labels/    多组结果对比（比如不同配置各跑一遍）
    if os.path.isdir(os.path.join(args.results, "labels")):
        groups = [""]
    else:
        groups = sorted(d for d in os.listdir(args.results)
                        if os.path.isdir(os.path.join(args.results, d, "labels")))
    if not groups:
        print(f"{args.results} 下没找到 labels 目录")
        return

    rows = []
    print(f"\nIoU 阈值：{args.iou}\n")
    header = f"{'结果':<16}{'图数':>6}{'真值框':>8}{'预测框':>8}{'命中':>7}{'漏检':>7}{'误检':>7}{'类别错':>8}{'召回率':>9}{'精确率':>9}{'F1':>8}"
    print(header)
    print("-" * len(header) * 2)

    if args.only:
        groups = [args.only]

    for s in groups:
        labels_dir = os.path.join(args.results, s, "labels") if s else os.path.join(args.results, "labels")
        if not os.path.isdir(labels_dir):
            continue
        stats, per_class, confusions = evaluate(args.gt, labels_dir, args.iou)
        rows.append({"group": s or "results", **stats})
        print(f"{(s or 'results'):<16}{stats['images']:>6}{stats['gt_boxes']:>8}{stats['pred_boxes']:>8}"
              f"{stats['tp']:>7}{stats['missed']:>7}{stats['false_pos']:>7}{stats['wrong_class']:>8}"
              f"{stats['recall']:>9.2%}{stats['precision']:>9.2%}{stats['f1']:>8.2%}")

        # 每个策略单独导一份按类别的明细
        pc_path = os.path.join(args.results, f"eval_per_class{('_' + s) if s else ''}.csv")
        with open(pc_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "真值数", "命中", "漏检", "误检", "召回率"])
            for cid in sorted(per_class):
                d = per_class[cid]
                r = d["tp"] / d["gt"] if d["gt"] else 0
                w.writerow([cid, d["gt"], d["tp"], d["missed"], d["fp"], f"{r:.2%}"])

        if confusions:
            top = sorted(confusions.items(), key=lambda x: -x[1])[:5]
            pairs = "，".join(f"真值{g}→预测{p}({n}次)" for (g, p), n in top)
            print(f"{'':<16}最常见的类别混淆：{pairs}")

    out = os.path.join(args.results, "eval_summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n汇总已保存：{out}")
    print("\n怎么读这些数字：")
    print("  召回率 = 该框的框出来了多少。预标注场景这个更重要 —— 漏的要人工补画。")
    print("  精确率 = 框出来的有多少是对的。低了只是多点删除操作，代价小得多。")
    print("  类别错 = 位置框对了但认错了类，这类最隐蔽，人工复核时容易漏掉。")


if __name__ == "__main__":
    main()
