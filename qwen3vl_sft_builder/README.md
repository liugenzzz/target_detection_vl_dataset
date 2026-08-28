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

## 四步走

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

## 指代短语不能泄漏答案

第一轮问的是「图中{referring}是什么？」，答案是类别名。**指代短语里绝不能出现
类别名**，否则等于把答案写进了问题：

```
问：图中靠近树木的那辆三轮车是什么？      答：是tricycle。
```

模型学不到任何识别能力，只学会把中文类别名翻译成英文。实测这类样本一度占到
VLM 指代的一半 —— 用类别名指代是最自然的说法，不明令禁止模型一定会那么写。

两道防线：

1. `prompts/vlm_describe.txt` 里硬禁止，并给出正反例。只写「优先用外观特征」
   不够，必须写「绝对不能出现类别名」并举反例。
2. `core/referring.py` 的 `leaks_label()` 兜底。**提示词管不住模型，代码这一层必须拦。**
   命中就丢弃该指代、回落模板空间指代 —— 宁可指代弱一点，也不能泄漏答案。
   拦截次数记在构建报告的 `vlm_referring_leaked_label`。

### 三种泄漏形式

**① 字面类别名** —— 「靠近树木的那辆**三轮车**」。检测只对中文类别名生效
（英文类别名和中文指代没法比对），三档判据：

| 判据 | 例子 |
|---|---|
| 类别名整体出现 | 「靠近树木的那辆**三轮车**」/ 三轮车 |
| 两字以上尾缀出现 | 「穿红色上衣的那个**人员**」/ 军事人员 |
| 单字尾缀出现在**结尾** | 「画面中部那艘**船**」/ 其它辅助船 |

第三档必须限定在结尾，否则「车」「船」这类高频字会大量误判 ——
「停在斜坡上、**车**头朝左的那个目标」并没有泄漏「卡车」。

**② 部件名和动作词**（隐蔽形式，比字面泄漏更难发现）：

```
车身银色的那个              「车身」把答案限定成车辆，砍掉人员、船舶
一名穿粉衣的人正骑行的那个    「骑行」限定成自行车/摩托车/三轮车
停在大石头后方的那个         「停」暗示是可停放的载具
```

词表在 `config` 的 `quality.category_hint_words`（默认 33 个，含部件名与动作词），
**按你的类别体系增删**。指代只该用**位置 + 颜色 + 与周围物体的相对关系**，
这三类都不暗示类别。

**③ 模板兜底自己泄漏** —— 这是曾经的一个严重 bug。模板指代在分区内不唯一时
会拼「那个{label}」来消歧，生成出「图中上方右侧那个van是什么？」。更糟的是
检测到 VLM 指代泄漏后回落的正是这个模板，等于原地打转。现在模板**绝不带类别名**，
分区内不唯一而 VLM 又没给出可用指代时，**整条样本丢弃**（计入
`dropped_ambiguous_referring`）—— 有歧义的问题是坏数据，但靠泄漏答案来消歧更糟。

指代里提到**其他类别**的物体是允许的（「白色轿车旁边的那个」「戴红帽子的人左边的那个」），
那是地标式指代 —— 用别的东西定位目标，没有暴露目标自己是什么，正是鼓励的写法。

### 别用「目标」

「目标」是标注术语，正常人不会问「那个目标是什么」。收尾称呼由
`quality.neutral_noun` 控制，默认光杆的「那个」：

```
图中画面右下角、银色的那个是什么？        自然、中性
图中画面右下角、银色的那个目标是什么？     标注腔
```

不用「物体」是因为类别里有 `人员`，把人叫「那个物体」很别扭。

## 指代短语要短

指代的唯一职责是**把这个目标从图中锁定出来**，能区分就够了。模型倾向于把看到的
特征全堆上去，实测会写出：

```
图中画面中上部，停在白色轿车和黑色轿车之间，靠近一棵绿树的那个目标是什么？
```

三从句 30 字，而它只是问题的主语。多目标样本更糟 —— 三个这样的指代拼起来接近
80 字，问题读不下去，训练时真正的信号也被淹没在定语里。

三道约束：

1. **提示词**要求「越短越好，不超过 15 字，优先只用一个最显眼的特征」，
   并给出「太长」的反例。只给字数上限不够，模型会正好写满上限。
2. **代码兜底** `too_long()`，超过 `quality.max_referring_len`（默认 20 字）
   就丢弃，回落模板空间指代 —— 模板虽短但足够锁定，强过读不下去的长句。
   拦截次数记在报告的 `vlm_referring_too_long`。
3. **多目标样本优先挑显眼、好描述的目标**（`easy` 档：大、孤立、分区内唯一），
   这类目标一两个特征就能锁定，指代自然短；拼起来仍超长的整条丢弃，
   这些目标各自出单目标样本。

实测效果：问题长度中位 17 字、最长 29 字。

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
| `turn1_identify.txt` / `turn1_answer.txt` | 第一轮：指代锁定 + 类别 |
| `turn2_locate.txt` | 第二轮：要位置 |
| `turn3_describe.txt` | 第三轮：要描述 |
| `multi_turn*.txt` | 多目标样本的三轮 |
| `negative_ask.txt` / `negative_answer.txt` | 拒答样本 |
| `inv_ask_*.txt` / `inv_answer_what.txt` | 盘点 → 定位 → 描述三轮的问法池 |
| `detect_class.txt` | 按类别全找出来的问法池 |
| `gen_phrases.txt` | 扩充问法库用的提示词，见下节 |
| `vlm_select.txt` | **调 VLM 的主提示词，最需要反复调的就是这个** |

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
              difficulty 难度分级与配额 / grouping 来源分组 / referring 指代生成
              vlm_client 调 Qwen 服务 / builder 三轮样本组装 / pipeline 编排
              phrase_bank 扩充问法库的读写与校验
scripts/      check_vlm.py 服务自检 / analyze.py 分布分析
              build.py 构建 / preview.py 验证图
              build_measure_words.py 量词表 / build_phrase_banks.py 扩充问法库
              get_visdrone.py 下载测试数据集
tests/        回归测试，不依赖外部数据和 VLM 服务
```

```bash
python tests/test_pipeline.py      # 39 项，服务器上部署后先跑这个
```

---

## 参考实现

坐标换算与空间指代思路来自 `qweb3vl_grouding_vqa_lp_gai`，类别表双向校验思路
来自 `vlm-bbox-labeling`。两者是**参考**，本项目不 import 它们。
