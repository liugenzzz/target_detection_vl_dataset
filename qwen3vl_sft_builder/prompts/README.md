# 提示词索引

全部是纯文本 `.txt`，与代码分离。改提示词不用动代码、不用重装依赖，
服务器上直接改文件即可。加载按**文件名**查找，所以怎么分目录都不影响调用点；
文件名全局唯一，重名会在加载时立刻报错。

## 按任务分（改这些只影响对应的那一个任务）

| 目录 | 任务 | 文件 | 作用 |
|---|---|---|---|
| `ground_unique/` | 🔷主线 | `ground_unique.txt` | 「定位图中的卡车。」问法池 |
| `ground_attribute/` | 🔷主线 | `ground_attribute.txt` | 兜底问法池（正常走 VLM 现场生成） |
| `inventory_locate/` | 🔷主线 | `inv_ask_what.txt` | 第一轮：图中有什么 |
| | | `inv_answer_what.txt` | 第一轮答案：清单 |
| | | `inv_ask_box.txt` | 第二轮：要那个目标的框 |
| `detect_class/` | | `detect_class.txt` | 「框出图中所有的人员。」问法池 |
| `region_identify/` | | `region_identify.txt` | 「[框] 区域内的是什么？」 |
| | | `region_identify_answer.txt` | 答案：说出类别 |
| `exist_negative/` | | `exist_ask.txt` | 存在性提问（中立，不能带预设） |
| | | `exist_yes_answer.txt` | 答「有」并报数 |
| | | `exist_yes_vague.txt` | 答「有」不报数（该类有实例被过滤时） |
| | | `exist_no_answer.txt` | 拒答 |
| `attribute_qa/` | | `attribute_qa_color.txt` | 问颜色 |
| | | `attribute_qa_feature.txt` | 问外观特征 |
| `spatial_relation/` | | `spatial_ask_lr.txt` | 左右关系专用（只有这个能用「哪一侧」） |
| | | `spatial_ask_ud.txt` | 上下关系专用 |
| | | `spatial_answer.txt` | 答案 |

## `describe/` —— 七种描述子类型，各一个文件

主线最后一轮的描述分七种，**每种的答案信息结构完全不同**（不是换说法）。
一个文件一种，改哪种就编哪个文件；不想要就在 config 里把
`tasks.ground_<子类型>` 的权重设 0。

| 文件 | 对应任务 | 答案里装什么 | 明确不装什么 |
|---|---|---|---|
| `appearance.txt` | `ground_appearance` | 只说物体本身：颜色、形状、部件、载物，至少三项 | 一个字不提方位和周边 |
| `state.txt` | `ground_state` | 只说状态动作：停着 / 行驶 / 装载 / 门开着 | 不说颜色、不说方位 |
| `part.txt` | `ground_part` | 聚焦一个部位展开 | 不铺开说整体 |
| `position.txt` | `ground_position` | 只说方位，可以说得很细 | 不说外观 |
| `relation.txt` | `ground_relation` | 只说和周围什么挨着 | 不说它自己长什么样 |
| `contrast.txt` | `ground_contrast` | 和图中同类比有什么不同 | — |
| `full.txt` | `ground_full` | 三段式（占七种的 20%） | — |

每个文件的格式：

```
#! kind: appearance                     子类型 id，对应任务名 ground_appearance
#! needs: 目标够大，能看清颜色或部件      适用条件，写进提示词让模型据此判断能不能做
#! must-not: 位于 画面 方位 旁边 周围     答案里不该出现的词，越界整条丢
#! max-grade: medium                    适用的最难档位，可省，省了表示不限
answer-spec: 只写这个物体自己身上的…      答案该装什么（可跨行，续行缩进）
q-example: 这辆三轮车本身长什么样？
a-example: 深红色车身，支着一顶白色遮阳篷…
```

**`must-not` 是七种能不能立住的关键。** 模型很容易把「只说外观」写成
「位于画面左侧的一辆红色三轮车」—— 加了方位就又滑回三段式。没有这道闸，
七种跑几轮会退化成同一种。

**`max-grade` 卡的是难度档位。** `part`（聚焦部位）和 `contrast`（同类对比）
要求看清细部，派给困难目标（小、糊、密集）模型只能编一个部位出来 ——
而编造从答案文本上看不出来，`must-not` 那道闸也拦不住。所以这两种写
`#! max-grade: medium`：指派阶段跳过没有目标够档的图，生成阶段再兜一次底。
粗粒度的几种（`appearance`/`position`/`relation`）不设上限，否则困难目标
一条描述都出不来。

加一种子类型只要往这个目录丢一个文件，再去 `config/default.yaml` 的
`tasks` 里加一行权重 —— 代码不用动，回归测试会检查两边对齐。

---

## 共用件（改这些会影响多个任务，目录名以 `_` 开头作提示）

| 目录 | 文件 | 影响谁 |
|---|---|---|
| `_shared/` | `ask_describe.txt` | 描述问句的兜底池（正常走 VLM 按子类型现场生成） |
| | `short_answer_suffix.txt` | 全部任务的短答案后缀 |
| `_vlm/` | `vlm_select.txt` | **最需要反复调的一个** —— 挑对象 + 属性 + 定位问句 + 按指派的子类型写描述，一张图一次调用 |
| `_tools/` | `gen_phrases.txt` | `build_phrase_banks.py` 扩充问法库 |
| | `measure_words.txt` | `build_measure_words.py` 生成量词表 |
| | `review.txt` / `review_sample.txt` | `review.py` 全量质检 |

## 问法池的写法

任务目录下的文件多数是**问法池**：每个非空非注释行是一种写法，构建时随机取一句。
文件开头的 `#` 注释说明这个池子干什么用，`#!` 开头的是校验指令：

```
#! forbid: 在哪 位置 坐标        这些词不能出现（语义闸：防止「描述」池混进「问位置」）
#! require-any: 所有 全部        必须含其中一个（防止 detect_class 退化成问单个）
#! optional-group: mw label      这几个占位符可以整组省略，但不能只省一半
```

这些指令同时用于**过滤扩充问法库的生成结果**和**装载时的再校验**，
所以手改这些文件时改坏了会被拦下来并报出丢弃数，而不是静默带进十万条数据。
