# Qwen3-VL 视觉定位 SFT 语料构建

把 YOLO 格式的目标检测标注，构建成 **Qwen3-VL-8B-Instruct** 可直接微调的
**多轮 ShareGPT 语料**。坐标输出 `0~1000` 归一化的 `bbox_2d`。

本项目**独立自持**，不依赖仓库里的其他项目，整个目录拷到服务器即可运行。

---

## 部署（远程 Linux 服务器）

```bash
pip install -r requirements.txt

cp config/local.yaml.example config/local.yaml
vim config/local.yaml          # 只改这个文件，不动代码，不动 default.yaml
```

`local.yaml` 里至少要填四项路径。**Windows 路径用单引号** —— YAML 的双引号
字符串会把反斜杠当转义符，`"F:\AI-Haishi\..."` 会报 `unknown escape character 'A'`：

```yaml
paths:
  # Linux
  labels_dir:   '/mnt/data2/.../dataset_det/labels/train'
  images_dir:   '/mnt/data2/.../dataset_det/images/train'
  classes_yaml: '/mnt/data2/.../data.yaml'        # 347 类那份
  output_dir:   './output'

  # Windows —— 单引号（推荐）、正斜杠、双反斜杠三种都行
  # labels_dir: 'F:\AI-Haishi\project\labels\train'
  # labels_dir: "F:/AI-Haishi/project/labels/train"
  # labels_dir: "F:\\AI-Haishi\\project\\labels\\train"
```

密钥不要写进文件，用环境变量注入：

```bash
export VLM_API_KEY=...
export VLM_API_URL=http://192.168.78.36:3012/v1/chat/completions
```

支持的环境变量：`VLM_API_KEY` `VLM_API_URL` `VLM_MODEL`
`LABELS_DIR` `IMAGES_DIR` `CLASSES_YAML` `OUTPUT_DIR`。

配置优先级：`default.yaml` < `local.yaml` < `--config` < 环境变量。

---

## 测试数据

本项目的测试结果基于两个公开数据集：

| 数据集 | 规模 | 位置 | 作用 |
|---|---|---|---|
| **COCO128** | 128 图 / 929 框 / 80 类 | 仓库自带 `vlm-bbox-labeling/coco128/` | 日常稀疏场景，每图均 7.4 个目标 |
| **VisDrone2019-DET-val** | 548 图 / 38759 框 / 10 类 | `python scripts/get_visdrone.py` 下载 | 无人机航拍、小目标密集，每图均 70 个目标 |

**主要用 VisDrone**：它是航拍视角、小目标（短边中位数仅 18px）、人员车辆密集，
和业务数据的特征对得上。COCO 每图才 7 个目标，测不出密集场景的问题 ——
本项目「每图框数上限」那条规则就是错误设计，在 COCO 上完全看不出来，
换到 VisDrone 才暴露出它会跳过 88% 的图。

VisDrone 覆盖不到的：全是日间彩色图，**测不了夜视/红外场景**。

```bash
python scripts/get_visdrone.py --out ./data/visdrone   # 约 78MB
```

## 五步走

```bash
# 0. 接入新的模型服务后第一个跑这个，别跳过
python scripts/check_vlm.py

# 0.5 一次性生成量词表与扩充问法库，之后长期复用（都是纯文本调用，不发图片）
python scripts/build_measure_words.py
python scripts/build_phrase_banks.py

# 1. 先看分布，定阈值（COCO 的阈值不适用于你的数据，必须重跑）
python scripts/analyze.py

# 2. 小批量试跑，人工看验证图
python scripts/build.py --limit 200
python scripts/preview.py --jsonl output/train.jsonl -n 60

# 3. 确认没问题再全量
python scripts/build.py

# 4. 全量质检：每条问答对让大模型对着原图核对一遍并打分
python scripts/review.py

# 5. 数据集体检：不调模型，纯离线的客观指标（几秒钟）
python scripts/dataset_stats.py
```

产出：

```
output/
├── train.jsonl          训练集
├── val.jsonl            验证集
├── build_report.json    构建报告：难度分布、配比、易混类别、划分校验
├── verify/              验证图 + manifest.json（人工复核看这个）
└── vlm_cache/           VLM 结果缓存，支持断点续跑
```

---

## 样本结构

一条样本 = 一张图 + 一个目标 + 三轮递进对话：

| 轮次 | 问 | 答 |
|---|---|---|
| 1 识别 | 图中**中部右侧那个目标**是什么？ | 是人员。 |
| 2 定位 | 请给出它在图中的位置。 | `{"bbox_2d":[718,331,742,379],"label":"人员"}` |
| 3 描述 | 描述一下这个人员。 | *（VLM 生成的自然语言描述）* |

**第一轮为什么用指代锁定，而不是直接问「图中是什么」**：一张图里有多个目标时，
直接问「图中是什么」却只答一个，等于在教模型漏报（实测 VisDrone 每图均 70 个目标）。
用指代先锁定对象，三轮都指向同一个目标，每轮答案都是真话。

另有两类变体，占比在 config 里配：

- **多目标**（默认 10%）：一次涉及 2~3 个目标，第二轮答案是 JSON 数组。
- **拒答**（默认 5%）：问图中不存在的类别，答「图中没有 X」。
  **没有这类样本，模型会学到「被问就一定有」的先验，推理时凭空编框。**

### 格式硬约束

`<image>` 占位符**只在第一轮 human 出现一次**。多轮里每轮都加会导致图像 token
重复注入，训练直接崩。`validate_sample()` 强制检查这一条，不合格的样本会被丢弃
并计入报告的 `invalid_dropped`。

---

## 质量过滤与难度配额

### 尺寸阈值用面积占比，不用绝对像素

业务数据分辨率跨度 `128x128 ~ 2048x1440`。同一个目标（占图宽 6.6%、高 2.8%）
在 128px 原图上只有 3.6px、在 2048px 原图上有 40.8px —— **绝对像素阈值会对
同一个目标给出相反判定**。而模型会把图缩放到自身输入分辨率，实际看到多少像素
只由面积占比决定（上例中恒为 44px）。

配置里的 `size_*_px` 是「模型实际看到的等效像素」，由面积占比换算而来：

| 档位 | 等效像素 | 对应面积占比 |
|---|---|---|
| easy | ≥ 64px | ≥ 0.391% |
| medium | ≥ 32px | ≥ 0.098% |
| hard | ≥ 16px | ≥ 0.024% |
| 剔除 | < 16px | < 0.024% |

### 难度取三个维度里最难的一项

- **尺寸**：等效像素（见上）
- **密集度**：同图同类目标数（>8 个判定无法可靠指代，剔除）
- **指代难度**：3×3 分区内同类是否唯一 —— 唯一则模板空间指代可用，否则需 VLM 消歧

### 配额必须全局做，不能逐图做

数据里存在大量整张图全是困难目标的图，逐图配额时只能全取困难的。
**实测逐图配额得到 40.7% 困难目标，改成两阶段后精确命中 10%**：

1. 逐图 `pick_candidates()`：简单档优先，取满 `samples_per_image_cap`
2. 全局 `balance_hard_quota()`：按 `hard / (easy+hard) = quota` 下采样困难档

### 没有「每图框数上限」这条规则

曾经有，是设计错误，已废弃。航拍/密集场景天生每图几十上百个目标，但
**「框多」本身不是质量问题** —— 一张有 70 辆车的停车场航拍图，每辆车都标得很准。
实测在 VisDrone 上启用「≤20 框/图」会跳过 **480/548 张图（88%）**。
密集问题交给难度分级**逐目标**判断，比整张图一刀切精细得多。
`sanity_max_boxes`（默认 1000）只用于拦截损坏的标注文件。

---

## 穷举式问句按【过滤前】的标注把关

「定位图中的卡车」「框出图中所有的人员」这类问句隐含着「答案是完整的」。
质量过滤会丢掉太小、太模糊的框，于是过滤后看着唯一的类别，原图里可能有好几个：

    原始标注   4 辆三轮车
    过滤后     1 辆（3 辆太小被丢掉）
    问         定位图中的三轮车。
    答         [一个框]                  ← 在教模型漏检

实测这种情况占 `ground_unique` 可选组合的 **45.2%**。所以这几个任务改成按
`Ctx.all_kept(label)`（该类过滤前后框数一致）把关：

| 任务 | 收紧成 |
|---|---|
| `ground_unique` | 该类在**原始标注**里也只有一个实例 |
| `detect_class` | 该类**全部框**都通过了质量过滤 |
| `spatial_relation` | 参照物那一类在原始标注里也唯一，否则「在卡车的哪一侧」指不明确 |
| `exist_negative` | 该类有实例被过滤时只答「有，图中存在X」，不报数 —— 报出来的数一定偏小 |

`inventory_locate` 不在此列：它第一轮问的是「图中有哪些**清晰可见**的目标」，
清单本来就限定在过滤后的集合上，问句和答案是自洽的。

实测代价比预想小得多（VisDrone 200 图，八任务全开）：

| | 样本总数 | 主线占比 | ground_unique | detect_class |
|---|---|---|---|---|
| 收紧前 | 896 | 39.8% | 4.8% | 8.8% |
| 收紧后 | 900 | 42.0% | 3.7% | 5.0% |

总量没掉，主线占比反而升了 —— 空出来的槽位被别的主线任务接走了。

这两个任务在 VisDrone 上都远低于配比里给的 12%，那是航拍密集小目标的特性
（每图均 70 个目标，绝大多数过不了质量过滤），不是收紧造成的：收紧前也只有
4.8% / 8.8%。换成实际数据（347 类、128×128~2048×1440）过滤会轻得多。

---

## 全量质检

构建阶段的过滤全是**结构性**的 —— 占位符对不对、有没有泄漏类别名、描述够不够长、
框有没有超出画面。这些都不看图。真正只有看图才能发现的问题，结构过滤一个也拦不住：

| 问题 | 为什么结构过滤拦不住 |
|---|---|
| 框偏了 | 坐标合法、格式合法，但框住的是旁边那棵树 |
| 编参照物 | 「旁边有一辆白色轿车」—— 图里根本没有轿车 |
| 指代不清 | 图里五辆一模一样的三轮车，问句只说「那辆三轮车」 |
| 类别认错 | 标注文件写的是卡车，图里其实是公交车 |

`scripts/review.py` 把每条问答对连同原图发给模型，四个维度各打 1~5 分：

| 维度 | 看什么 |
|---|---|
| `correct` | 答案和图对不对得上：框的位置、类别、数量、颜色 |
| `grounded` | 描述里说的东西图里是不是真有（编造参照物直接给 1 分） |
| `clear` | 照着问句能不能唯一地找到那个目标；描述是否空泛 |
| `instruction` | 问句是不是指令口吻 |
| `needs_image` | 这道题必须看图才能答吗。「街景图里有没有潜水艇？」答「没有」不看图也知道，这种样本训不出东西 |

`needs_image` **不进综合分** —— 它衡量的是「这条样本有没有训练价值」，
不是「这条样本对不对」。一条不看图也能答对的拒答样本并没有错，只是没用；
该不该留由 `review.min_dimension` 单独卡，卡得比别的松。

**其余四个维度取最小值作为综合分，不取平均。** 一条描述编造了参照物（`grounded=1`）
但问句写得漂亮（`instruction=5`），平均下来还有 3 分多，照样进训练集。质检要看短板。

### 三件必须知道的事

**一、不通过的不直接删。** 落进 `*.rejected.jsonl`，人工扫一眼再决定 ——
审核模型自己也会看错。

**二、按图分组，一张图上的样本合并成一次调用。** 图片的 base64 是请求里最大的
一块，分开发等于把同一张图传 N 遍。十万条样本按每图 8 条算，分组后约一万两千次
调用，不分组是十万次。结果按图落盘缓存，中断了重跑接着走；改阈值重判用
`--dry-run`，一次请求都不发。

**三、别用生成它的同一个模型自审。** 它会倾向于认同自己的输出。
`vlm.roles.review` 可以给质检单独指定模型或整个模型池：

```yaml
vlm:
  roles:
    review:
      temperature: 0.0
      endpoints:
        - {name: "质检", api_url: "http://.../v1/chat/completions",
           model: "Qwen2.5-VL-72B", concurrency: 4}
```

质检的缓存自动落在 `<cache_dir>/review/` 子目录 —— 与构建的缓存键空间隔离，
换质检模型只想重跑质检时删一个目录即可。

---

## 数据集体检（客观指标）

`scripts/review.py` 是**主观**质检 —— 让大模型看图逐条打分。
`scripts/dataset_stats.py` 是**客观**体检 —— 不调模型，纯离线统计，几秒钟跑完。

两者互补，缺一不可。主观质检能发现「框住的是旁边那棵树」，客观体检发现不了；
客观体检能发现下面这些**分布层面**的问题，而它们单看任何一条样本都是好的，
主观质检逐条打分永远发现不了：

| 指标 | 对标 | 抓什么 |
|---|---|---|
| `CHAIR_i` / `CHAIR_s` | CHAIR (Rohrbach 2018) | 答案里提到的类别有多少不在该图标注里。原版比对 COCO 标注，我们比对 YOLO 标注文件，思路相同且更严格 —— 标注文件就是真值 |
| 类别覆盖率 + 基尼系数 | 长尾分析 | 347 个类别只练到 40 个，训出来的模型就只认那 40 个 |
| 九宫格框分布 | — | 85% 的框挤在画面中央的话，模型学到的是「往中间猜」 |
| `Distinct-2` / `Distinct-3` | Distinct-n (Li 2016) | 词汇多样性 |
| 答案开头集中度 | — | 描述全是「位于画面…」一个句式。带判据：起手方式由 `desc_opening.txt` 轮换 N 种，最集中的一种超过 `2/N` 就判不合格 |
| 正负比 + 难负样本率 | POPE (Li 2023) | 存在性问答失衡会让模型学成「一律答有」 |

两个刻意的取舍：

**CHAIR 只统计类别表里的词。** 描述里的「斑马线」「路灯杆」不在类别表中，
无从判定真假，不计入。这会**低估**真实幻觉率，但绝不会误报 —— 一个客观指标
出现假阳性比漏报更糟。

**Distinct-n 在固定 2000 条子样本上算。** 原始 Distinct-n 有语料规模偏差：
语料越大分母涨得比分子快，1 千条和 10 万条算出来必然是后者低，那是规模造成的、
不是多样性真的下降。固定子样本量之后跨版本才可比。

---

## 模型池

`vlm.endpoints` 列出多路，不写就把平铺的 `api_url`/`model` 当成只有一路，
老配置不用改。

- 总并发是各路 `concurrency` 之和，请求**按 `concurrency` 加权**轮转 ——
  一路写 8、一路写 3 就按 8:3 分。均分会把慢的那路压垮、快的那路闲着
- 每次重试换一路：一路 401、一路超时，轮着试还能跑完
- 配置类错误（401/404/模型名不对）只摘除**那一路**并打日志，其余照跑；
  全部端点都挂了才中止整批
- `check_vlm.py` 逐路自检 —— 只测第一路的话，第二路配错要等跑全量才暴露

缓存键不含模型名，池子里应该放**可互换**的模型；换成能力明显不同的模型时
顺手换个 `cache_dir`。

---

## 跨任务一致性

同一张图会被抽出多条样本、分给不同任务。它们各自独立生成，但说的是同一张图：

```
inventory_locate  「图中有 3 名人员、2 辆卡车。」
detect_class      「框出图中所有的人员。」-> 必须正好 3 个框
exist_negative    「图中有没有直升机？」-> 没有 -> 别的样本里就不能出现直升机
```

任何一条对不上，就是同一张图配了两套真值，模型只能学成随机猜。

结构上这已经由「一张图只有一份过滤后框集合喂给全部任务」保证 —— 曾经因为
`clean_labels` 按任务重算可用类别，同一张图上一个样本说没有人员、另一个又去
定位人员。`core/consistency.py` 每次构建核对一遍，结果进 `build_report.json`：

- 盘点的数量 vs `detect_class` 的框数 vs `exist_yes` 报的数
- 答「没有X」的样本 vs 别的样本框出的 X
- 任何样本给出的框数不得超过过滤后实际有的

VisDrone 200 图、854 条样本实测 0 冲突。

---

## train/val 必须按来源分组划分

**不能按图片随机划分。** 业务数据的文件名形如：

```
Wasserfalle-...-Hochwasser_mp4-14_jpg.rf.062732212bcc....txt
Wasserfalle-...-Hochwasser_mp4-15_jpg.rf.b1db71c78b1a....txt
y2mate_com-...-Rio-Cesar_480p_mp4-8_jpg.rf.88ec9a5c58ac....txt
y2mate_com-...-Rio-Cesar_480p_mp4-8_jpg.rf.b97e1a6cca79....txt
```

暴露两层重复：`.rf.<hash>` 是 Roboflow 对**同一张原图**做不同增强导出的多个文件；
`_mp4-<帧号>` 说明图片来自**视频抽帧**，相邻帧画面几乎一致。

按图片随机划分会把同源图拆到 train 和 val 两边，**验证集里全是训练时见过的画面，
指标虚高且看不出来**。`core/grouping.py` 把文件名归约到原始来源，同组整组进
train 或整组进 val。构建报告里的 `split.group_overlap` 必须为 0。

---

## 接入模型服务

`scripts/check_vlm.py` 三步递进地验证，每一步失败都会指出该改哪里：

| 步骤 | 验证 | 失败通常意味着 |
|---|---|---|
| 1 纯文本 | 地址 / 端口 / model 名 / api_key | 见下表 |
| 2 图片输入 | 这个部署开没开多模态 | 部署没开视觉，或不认 `image_url` 写法 |
| 3 真实提示词 | 返回能不能解析成 JSON | 改 `prompts/vlm_describe.txt`，不用动代码 |

常见状态码的含义与处置：

| 状态码 | 含义 | 怎么改 | 会重试吗 |
|---|---|---|---|
| 401 | api_key 没设或不对 | 设 `VLM_API_KEY` 或填 `vlm.api_key` | 否，立即中止 |
| 403 | key 对但无该模型权限 | 找服务方确认 | 否，立即中止 |
| 404 | 路径不对 | `api_url` 要带完整的 `/v1/chat/completions` | 否，立即中止 |
| 400 / 422 | 模型名对不上，或不支持图片 | 改 `vlm.model` | 否，立即中止 |
| 429 | 限流 | 调小 `vlm.concurrency` | 是 |
| 5xx | 服务端故障 | 等或重启服务 | 是 |

配置类错误（4xx，429 除外）**不重试也不继续** —— 这类错误必然对所有请求成立，
十万条任务会变成三十万次注定失败的请求。检测到就立刻中止并打印该怎么改。

第 2 步还会报出**单次图片请求耗时**，用它估算并发数 —— 这是定 `vlm.concurrency` 的唯一依据。

接口按 **OpenAI 兼容格式** 调用（`/v1/chat/completions` + `image_url` 传 base64），
vLLM、Ollama、LM Studio、SGLang 的兼容层都支持。本地常见地址：

```yaml
vlm:
  api_url: "http://localhost:11434/v1/chat/completions"   # Ollama
  api_url: "http://localhost:8000/v1/chat/completions"    # vLLM / SGLang
  api_url: "http://localhost:1234/v1/chat/completions"    # LM Studio
```

## 描述语句

第三轮的描述由 `vlm.api_url` 指向的自建 Qwen 服务生成，**同一次调用还会返回
视觉指代短语**，所以给同类密集目标消歧不额外增加成本。

**`vlm.enabled: false` 时全部回落到模板描述**（形如「中部右侧是一个人员。」）。
模板只陈述标注本身可确定的事实，绝不编造外观属性 —— 但它非常干瘪，
**正式构建必须开启 VLM**，模板只是本地调试和单条失败时的兜底。

VLM 结果按图落盘到 `vlm_cache/`，**支持断点续跑** —— 两万张图跑一遍要几小时，
中途挂掉从头再来是不可接受的。

---

## 提示词

全部在 `prompts/` 下，是纯文本 `.txt`，与代码分离。改提示词不用动代码、
不用重装依赖，服务器上直接改文件即可。

| 文件 | 用途 |
|---|---|
| `ground_unique.txt` / `ground_attribute.txt` | 主线的定位问句 |
| `inv_ask_*.txt` / `inv_answer_what.txt` | 盘点 → 定位 → 描述三轮 |
| `ask_describe.txt` | 主线最后一轮：要描述 |
| `detect_class.txt` | 按类别全找出来 |
| `spatial_ask_lr/ud.txt` / `spatial_answer.txt` | 方位关系，左右和上下分开 |
| `attribute_qa_color/feature.txt` | 属性问答的两个维度 |
| `exist_ask.txt` / `exist_*_answer.txt` | 存在性问答与拒答 |
| `region_identify*.txt` | REG：给框问类别 |
| `desc_opening.txt` | 描述的起手方式，每张图轮换一条 —— 见下 |
| `gen_phrases.txt` | 扩充问法库用的提示词 |
| `review.txt` / `review_sample.txt` | 全量质检 |
| `measure_words.txt` | 一次性生成量词表 |
| `inv_ask_*.txt` / `inv_answer_what.txt` | 盘点 → 定位 → 描述三轮的问法池 |
| `detect_class.txt` | 按类别全找出来的问法池 |
| `gen_phrases.txt` | 扩充问法库用的提示词，见下节 |
| `vlm_select.txt` | **调 VLM 的主提示词，最需要反复调的就是这个** |

### 描述的起手方式要轮换

提示词里给固定的例子，模型会朝那几个例子的句式收敛 —— 例子越少收敛得越死
（从 1 个例子加到 3 个，描述去重率从 79/142 涨到 111/112）。但只要例子是固定的，
收敛就只是被削弱、没有消除：十万条描述可能全是「位于画面X角，一辆……」。

`prompts/desc_opening.txt` 是一个「起手方式 ||| 示例」的池子，**每张图随机抽一条**
塞进 `vlm_select` 的提示词，等于把「例子」这个变量本身也随机化了。

效果**必须在真实服务上验**，用 `scripts/dataset_stats.py` 的「答案开头集中度」：
轮换 N 种，最集中的一种超过 `2/N` 就是模型没照做、退回了自己的套路。
那时把 `desc_opening.txt` 里的要求写得更硬，或者换描述能力更强的模型。

### 问法池与扩充问法库

上表里带「问法池」的文件，每个非空非注释行是一种说法，构建时随机取一句。
手写只有五六条，**十万条样本摊下来同一句话要出现上千遍** —— 模型学到的会是
「见到这句口令就输出框」，而不是「听懂要框什么」。

`scripts/build_phrase_banks.py` 把每个池子扩到几十条，结果落进
`config/phrase_banks.yaml`，构建时与手写的那几条合并取样。跟量词表一样，
问法与图片内容无关，所以一次性生成、长期复用，不占构建时的调用预算
（全部池子加起来十几次纯文本调用）。

```bash
python scripts/build_phrase_banks.py                      # 全部池子
python scripts/build_phrase_banks.py --pool inv_ask_what --show
python scripts/build_phrase_banks.py --target 60 --force  # 重新生成
```

### 问法是【指令】，不是聊天

这些句子是下达给视觉模型的指令，参照 RefCOCO / Qwen grounding 训练数据的口吻：
祈使句（「框出图中的卡车。」）或直接的疑问句（「卡车在图中的什么位置？」）。
**不要语气词、口头语、客套、儿化音** ——「诶那辆卡车在哪儿？」「帮我把人员框出来」
这类闲聊腔一律不收。花样靠换动词（定位/框出/标出/输出/找出/圈出）、
换要什么（位置/坐标/边界框/检测框）、换句式、换语序来变，不靠加语气词凑。

`config` 里的 `phrase_banks.forbid_global` 是这道语体闸，对所有池子生效。

### 四道闸

生成的句子入库前逐条校验，不合格直接丢，不做修补：

| 闸 | 拦什么 | 声明位置 |
|---|---|---|
| 占位符 | 少了问句失去指向（「那辆在哪？」），多了构建时 `.format()` 抛异常打断整批 | 从 `.txt` 的种子自动推出 |
| 语体 | 语气词、口头语、客套、儿化音 | `config` 的 `forbid_global` |
| 语义 | 「描述」池混进「问位置」；`ground_unique` 混进「所有的」（那是 `detect_class` 的意思） | `.txt` 里的 `#! forbid:` / `#! require-any:` |
| 句法 | 模型的元话语。「3. 这条带了序号」剥掉序号后结构完全合法，实测被当成第二轮问句用了 | 代码内置：必须有句末标点、不含元话语词 |

`#! optional-refer` 声明这一轮的指代能由上文承接（「描述该区域的内容。」紧跟在
刚给出框的那一轮后面），此时允许整句不带占位符 —— 但不允许只带一半，
只剩「这{mw}」比干脆不提还糟。

超长的丢掉、重复的丢掉。
**生成完请扫一眼 `config/phrase_banks.yaml`，别扭的句子直接删掉那一行**，
删完不用重新生成。

构建报告里的 `question_variety` 就是用来盯这件事的：`ratio` 是不同问法占问句
总数的比例，`most_repeated` 是复现最多的那一句。本地实测扩充前 31.8%、最高一句复现 37 次，扩充后 67.8%、最高 9 次。

---

## 易混类别

`core/classes.py` 会自动检测名称相近的类别组，两条判据：包含关系
（`人员` ⊂ `一般人员` / `军事人员`）、等长且一字之差（`切管器` vs `切管机`、
`压接钳` vs `压管钳`）。

这类类别靠视觉难以可靠区分。**构建时不阻塞**（标注文件已指定 `class_id`，
直接查表取名即可），但会在样本 `metadata.confusable_class` 打标记，
训练后若这几类混淆严重，可据此快速定位到是哪批样本。全部易混组会列在构建报告里。

---

## 目录

```
config/       default.yaml（进版本控制） + local.yaml（服务器上改，已 gitignore）
prompts/      提示词纯文本，与代码分离
core/         classes 类别表与易混检测 / coords 坐标换算 / yolo 标注解析
              difficulty 难度分级与配额 / grouping 来源分组
              tasks 八个任务的样本生成 / pipeline 编排 / sample 格式契约
              vlm_client 调模型（含模型池）/ phrase_bank 扩充问法库
              register 语体闸 / consistency 跨任务一致性
              review 主观质检 / stats 客观体检
              referring 描述成色判定与空间措辞 / cli 脚本入口包装
scripts/      check_vlm.py 服务自检 / analyze.py 分布分析
              review.py 主观质检 / dataset_stats.py 客观体检
              build.py 构建 / preview.py 验证图
              build_measure_words.py 量词表 / build_phrase_banks.py 扩充问法库
              get_visdrone.py 下载测试数据集
tests/        回归测试，不依赖外部数据和 VLM 服务
```

```bash
python tests/test_pipeline.py      # 55 项，服务器上部署后先跑这个
```

---

## 参考实现

坐标换算与空间指代思路来自 `qweb3vl_grouding_vqa_lp_gai`，类别表双向校验思路
来自 `vlm-bbox-labeling`。两者是**参考**，本项目不 import 它们。
