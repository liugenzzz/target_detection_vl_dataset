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

## 共用件（改这些会影响多个任务，目录名以 `_` 开头作提示）

| 目录 | 文件 | 影响谁 |
|---|---|---|
| `_shared/` | `ask_describe.txt` | 主线三个任务的**最后一轮**（要描述） |
| | `short_answer_suffix.txt` | 全部任务的短答案后缀 |
| `_vlm/` | `vlm_select.txt` | **最需要反复调的一个** —— 挑对象 + 生成问句 + 描述，一张图一次调用 |
| | `desc_opening.txt` | 描述的起手方式，每张图轮换一条 |
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
