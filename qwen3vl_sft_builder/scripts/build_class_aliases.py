#!/usr/bin/env python3
"""为类别表生成别名表初稿：class_id -> 规范中文名，外加要丢弃的 class_id。

    python scripts/build_class_aliases.py                    # 生成初稿
    python scripts/build_class_aliases.py --no-translate     # 只合并去重，不调模型

生成后在 config 里指上它：

    paths:
      class_aliases: "config/class_aliases.yaml"

【为什么要这张表】几个数据集合起来的类别表有三种毛病，一张映射全治：

  同物多编号  Apple(7) / Apples(8) / apple(624) / apples(626) 是同一种东西。
              所有任务按【名称字符串】分组：不合并的话「图中有多少个 apple」
              只数其中一份、「框出所有 Apple」只给一份的框 —— 教模型漏检。
              而且大小写不同的那几个连上下位判据都绕过了（"Apple" in "apples"
              是 False），会生成「图里有 Apples，问有没有 apple -> 没有」这种
              错误的拒答样本。
  非目标类别  air / sky / bedroom / people 没有可指代的边界，进 drop 名单。
  中英混排    英文原名译成中文，问句和描述的语言统一；顺带把上下位判据救回来
              —— 子串判据在中文复合词上成立（人员 ⊂ 军事人员），在英文上是
              灾难（ear ⊂ bear / beard / earring / year）。

【标注文件一个字都不用改】映射作用在 class_id 上，.txt 照旧。

【生成的是初稿，一定要人工过一遍】译名和 drop 名单都在文件里，每行带英文
原名注释。译歪的直接改那一行，不用重新生成。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                    # noqa: E402
from config import load_config                    # noqa: E402
from core import progress                          # noqa: E402
from core.classes import load_class_table         # noqa: E402
from core.vlm_client import VlmClient             # noqa: E402

BATCH = 50          # 一次翻 50 个，太多模型会漏答

# ---------------------------------------------------------------- 丢弃名单
# 这些不是「可以框出来并指代」的目标。判据是：能不能回答「框出图中所有的 X」
# 并且答案是一组有边界、可计数的框。
DROP_SCENE = {
    "airport", "amusement park", "apartment", "apartment building", "aquarium",
    "attic", "auditorium", "backyard", "bakery", "balcony", "barn", "bathroom",
    "beach", "bedroom", "bus stop", "cafe", "cafeteria", "cemetery", "city",
    "classroom", "closet", "coffee shop", "courtyard", "deck", "dining room",
    "dock", "docks", "factory", "field", "forest", "garage", "garden", "gym",
    "hallway", "harbor", "hospital", "hotel", "hotel room", "kitchen", "lake",
    "library", "living room", "lobby", "lounge", "mall", "marina", "market",
    "meadow", "museum", "ocean", "office", "orchard", "palace", "pantry",
    "park", "parking lot", "pasture", "patio", "pizza shop", "pond", "porch",
    "pub", "restaurant", "restroom", "river", "road", "roadside", "roadway",
    "room", "salon", "school", "sea", "shop", "shopping center", "sidewalk",
    "skate park", "sky", "stadium", "stage", "station", "store", "street",
    "supermarket", "temple", "theater", "town", "tunnel", "village", "yard",
    "zoo", "desert", "island", "plain", "highway", "intersection", "crosswalk",
    "runway", "taxiway", "terminal", "train station", "basketball court",
    "tennis court", "swimming pool", "ground track field", "soccer ball field",
    "car_parking", "truck_parking", "goods_yard", "path", "walkway", "lawn",
    "hill", "hillside", "hilltop", "mountain side", "shore", "cliff",
}
DROP_MASS = {
    "air", "water", "dirt", "mud", "grass", "snow", "fog", "smoke", "ice",
    "sand", "gravel", "wool", "skin", "fur", "hay", "moss", "weeds", "rain",
    "steam", "flames", "fire", "blood", "liquid", "cream", "dough", "flour",
    "sugar", "salt", "oil", "honey", "broth", "gravy", "sauce", "pavement",
    "ground", "floor", "ceiling", "wall", "wallpaper", "carpet", "tiles",
    "bricks", "glaze", "icing", "frosting", "powder", "powdered sugar",
    "cement_concrete_pavement", "sea foam", "coral", "grease", "trash",
}
DROP_BODY = {
    "arm", "ear", "eye", "face", "feet", "finger", "foot", "hair", "hand",
    "head", "heel", "horn", "leg", "lip", "mane", "mouth", "neck", "nose",
    "paw", "tail", "teeth", "thumb", "tongue", "waist", "wrist", "beak",
    "beard", "mustache", "trunk", "wing", "spots",
}
# 集合名词与上位词：和具体类别共存时，计数与穷举定位必然自相矛盾
# （图里有 3 个 person，问「有多少 people」答什么都不对）。
DROP_COLLECTIVE = {
    "people", "men", "women", "boys", "girls", "children", "guests",
    "passengers", "spectators", "tourists", "visitors", "customers",
    "shoppers", "workers", "students", "officers", "policemen", "soldiers",
    "crowd", "audience", "team", "family", "couple", "herd",
    "animal", "vehicle", "food", "fruit", "vegetable", "device", "gadget",
    "instrument", "tool", "utensil", "appliance", "accessory", "garment",
    "clothes", "outfit", "beverage", "drink", "snack", "dessert", "topping",
    "ingredient", "seafood", "meat", "herb", "spice", "alcohol", "liquor",
    "produce", "groceries", "merchandise", "baked good", "sporting equipment",
    "office supplies", "toiletries", "silverware", "jewelry", "money",
    "breakfast", "dinner", "lunch", "meal", "wedding",
    "word", "letter", "number", "character", "symbol", "logo", "graffiti",
    "artwork", "drawing", "decoration", "figure",
}
DROP_SETS = (("场景/地点", DROP_SCENE), ("材质/不可数", DROP_MASS),
             ("身体部位", DROP_BODY), ("集合词/上位词", DROP_COLLECTIVE))


def norm(name: str) -> str:
    return re.sub(r"[\s_]+", " ", str(name).strip().lower())


def singular(name: str) -> str:
    """粗暴去复数，只为把 Apple/Apples 归到一组，不追求语言学正确。"""
    s = norm(name)
    for suf, rep in (("ies", "y"), ("shes", "sh"), ("ches", "ch"),
                     ("ses", "se"), ("xes", "x"), ("s", "")):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            return s[: -len(suf)] + rep
    return s


def concept_key(name: str) -> str:
    """同一个概念的归一键。大小写、空格/下划线、单复数都归到一起。"""
    return singular(name)


def drop_reason(name: str) -> str:
    """要丢就返回理由，不丢返回空串。单复数两种写法都查。"""
    for form in (norm(name), singular(name), singular(name) + "s"):
        for tag, s in DROP_SETS:
            if form in s:
                return tag
    return ""


def translate(client: VlmClient, names: List[str]) -> Dict[str, str]:
    """分批调模型翻译。失败的那批留空，后面按原名落盘，人工补。"""
    out: Dict[str, str] = {}
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        raw = client._post({
            "model": client.model,
            "messages": [{"role": "user",
                          "content": prompts.render("class_zh",
                                                    names="、".join(chunk))}],
            "temperature": 0.0,
            "max_tokens": 4096,
        })
        n = i // BATCH + 1
        total = (len(names) + BATCH - 1) // BATCH
        if not raw:
            print(f"  [{n}/{total}] 调用失败，这批保留英文原名")
            continue
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        try:
            got = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            got = {}
        hit = 0
        for k, v in got.items():
            key = norm(k)
            if key in {norm(c) for c in chunk} and str(v).strip():
                out[key] = str(v).strip()
                hit += 1
        print(f"  [{n}/{total}] {hit}/{len(chunk)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "class_aliases.yaml")
    ap.add_argument("--no-translate", action="store_true",
                    help="只合并去重和丢弃，不调模型，规范名用英文原名")
    ap.add_argument("--no-counts", action="store_true",
                    help="不统计每个类别有多少框（跳过扫标注目录，快一点）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 这里【刻意不用 table_from_config】—— 要的是没套别名表的原始类别表，
    # 否则第二次跑会在上一次的结果上再合并一遍。
    table = load_class_table(cfg.require("paths.classes_yaml"))
    id2name = dict(table.id2name)
    print(f"原始类别 {len(id2name)} 个\n")

    # ---- 1) 丢弃 ----
    drop: Dict[int, str] = {}
    by_reason = collections.Counter()
    for cid, name in id2name.items():
        why = drop_reason(name)
        if why:
            drop[cid] = why
            by_reason[why] += 1
    print(f"丢弃 {len(drop)} 个编号：")
    for tag, n in by_reason.most_common():
        print(f"    {tag:14} {n}")

    # ---- 2) 合并 ----
    keep = {cid: n for cid, n in id2name.items() if cid not in drop}
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for cid, name in keep.items():
        groups[concept_key(name)].append(cid)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n合并：{len(keep)} 个编号 -> {len(groups)} 个概念"
          f"（{len(multi)} 个概念有多个编号，冗余 {len(keep) - len(groups)} 个）")

    # 每个概念挑一个代表名送去翻译：优先小写、优先单数、优先短
    rep: Dict[str, str] = {}
    for key, cids in groups.items():
        rep[key] = min((keep[c] for c in cids),
                       key=lambda n: (n != n.lower(), len(n), n))

    # ---- 3) 翻译 ----
    zh: Dict[str, str] = {}
    if not args.no_translate:
        client = VlmClient(cfg)
        if not client.enabled:
            raise SystemExit("vlm.enabled 为 false，无法翻译；"
                             "只想合并去重就加 --no-translate")
        todo = sorted({rep[k] for k in groups if re.search(r"[A-Za-z]", rep[k])})
        print(f"\n翻译 {len(todo)} 个概念：")
        zh = translate(client, todo)

    # ---- 3.5) 数一下每个类别有多少框 ----
    # 【为了让人能真的把这份表过一遍】。1,168 个概念按字母序排，人只会从 A
    # 看到 C 就放弃；按框数从多到少排，看前 100 条就覆盖大部分数据，剩下的
    # 长尾译错了也影响不到几个样本。没有这个排序，「人工过一遍」是句空话。
    counts: Dict[int, int] = collections.Counter()
    if not args.no_counts:
        labels_dir = Path(cfg.require("paths.labels_dir"))
        files = sorted(labels_dir.glob("*.txt"))
        bar = progress.make("统计类别频次", len(files), True)
        for f in files:
            bar.step()
            try:
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            counts[int(float(parts[0]))] += 1
                        except ValueError:
                            pass
            except OSError:
                continue
        bar.close()

    def group_boxes(key: str) -> int:
        return sum(counts.get(c, 0) for c in groups[key])

    # ---- 4) 落盘 ----
    lines = [
        "# 类别别名表。由 scripts/build_class_aliases.py 生成，**是初稿，要人工过一遍**。",
        "#",
        "# canonical: class_id -> 规范名。多个编号指向同一个名字 = 刻意合并，",
        "#            所有任务按名称分组，合并之后计数和穷举定位才对得上。",
        "# drop:      这些编号的框直接不进数据集（没有可指代的边界）。",
        "#",
        "# 译歪的直接改这个文件里那一行，不用重新生成。行尾注释是英文原名。",
        "",
        "canonical:",
    ]
    filled = 0
    # 框多的排前面 —— 人工校对从这里开始看
    for key in sorted(groups, key=lambda k: (-group_boxes(k), rep[k].lower())):
        cids = sorted(groups[key])
        n_box = group_boxes(key)
        name = zh.get(norm(rep[key]), "")
        if name:
            filled += 1
        else:
            name = rep[key]          # 没翻出来，先用原名占位
        for cid in cids:
            src = id2name[cid]
            note = f"  # {src}"
            if len(cids) > 1:
                note += f"  [{len(cids)} 合 1]"
            if not args.no_counts:
                note += f"  {n_box} 框"
            lines.append(f"  {cid}: \"{name}\"{note}")
    lines += ["", "drop:"]
    for cid in sorted(drop, key=lambda c: (-counts.get(c, 0), id2name[c].lower())):
        note = f"  # {id2name[cid]}  <- {drop[cid]}"
        if not args.no_counts:
            note += f"  {counts.get(cid, 0)} 框"
        lines.append(f"  - {cid}{note}")
    lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n写入 {args.out}")
    print(f"  规范类别 {len(groups)} 个（原 {len(id2name)}），丢弃 {len(drop)} 个编号")
    if not args.no_translate:
        miss = len(groups) - filled
        print(f"  译出 {filled} 个，{miss} 个没译出（文件里先用英文原名占位，需要手工补）")
    if not args.no_counts:
        ranked = sorted(groups, key=lambda k: -group_boxes(k))
        total_box = sum(counts.values())
        for top in (50, 100, 200):
            cov = sum(group_boxes(k) for k in ranked[:top])
            print(f"  按框数排序后，前 {top:>3} 个概念覆盖 {cov / max(total_box, 1) * 100:.1f}% 的框")
        dropped_box = sum(counts.get(c, 0) for c in drop)
        print(f"  drop 名单一共丢掉 {dropped_box:,} 个框"
              f"（占 {dropped_box / max(total_box, 1) * 100:.1f}%）")

    print("\n下一步：")
    print(f"  1. 打开 {args.out} 扫一遍，译歪的改掉，不想丢的从 drop 里删掉")
    print("  2. config 里加上 paths.class_aliases: \"config/class_aliases.yaml\"")
    print("  3. python scripts/analyze.py   看类别数和易混组是不是正常了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
