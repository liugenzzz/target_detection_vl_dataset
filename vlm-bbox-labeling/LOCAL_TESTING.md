# 本地测试文档

不用 Docker，在自己电脑上跑起来调试。Windows PowerShell 为例，Linux/Mac 把 `$env:X="v"` 换成 `export X=v` 即可。

---

## 第 0 步：装依赖

```powershell
pip install -r requirements.txt
```

---

## 第 1 步：先测后端模型（**别跳过**）

在起服务之前，先确认后端模型能不能用。这一步能提前发现绝大多数问题。

```powershell
python test_qwen.py 你的图片.jpg
```

脚本会依次验证三件事：

| 测试 | 验证什么 | 看什么 |
|---|---|---|
| 1. 纯文本连通性 | 地址、key、模型名对不对 | 通不过后面都别看了 |
| 2. 图片输入 + 画框 | 收不收图、给不给 JSON、**坐标是什么格式** | 最关键 |
| 3. logprobs | 支不支持置信度 | 决定 `ENABLE_LOGPROBS` |

**第 2 项的坐标格式判定决定了 `COORD_MODE` 怎么配**：

- 输出"0~1000 千分比坐标" → `COORD_MODE=per_mille`（Qwen3.6 实测就是这个）
- 输出"绝对像素坐标" → `COORD_MODE=pixel`
- 输出"0~1 相对坐标" → `COORD_MODE=relative`

如果第 2 步报错说不支持图片，说明该部署没开多模态输入，先解决这个。

---

## 第 2 步：把坐标画到图上确认

数值格式看着对，**不代表框的位置真的准**。把 test_qwen.py 输出的那段 JSON 存成 `boxes.json`（```json 包裹不用删），然后：

```powershell
python draw_boxes.py 你的图片.jpg boxes.json
```

生成 `你的图片_verify.jpg`，打开看：

- 框套在物体上 → 坐标模式对了，继续
- 框全挤在左上角一小块 → 其实是像素坐标，加 `--mode pixel` 重试
- 框整体偏移或超出画面 → 试 `--mode relative`

**这一步必须做。** 坐标模式选错的话，后面跑再多图都是废数据。

---

## 第 3 步：起服务

把类别表放好（项目根目录，命名 `classes.yaml`），然后：

```powershell
$env:CLASSES_YAML_PATH="./classes.yaml"
$env:QWEN_API_URL="http://192.168.78.36:3012/v1/chat/completions"
$env:QWEN_API_KEY="sk-bveYeVn6NAdRRElTWCqhtyJbkTL5XwweedczV9FJ05kDqhqX"
$env:COORD_MODE="per_mille"
$env:QWEN_MAX_TOKENS="3000"
$env:ENABLE_LOGPROBS="true"
$env:DRAW_INVALID="false"

uvicorn main:app --host 0.0.0.0 --port 8000
```

调试时加 `--reload`，改代码自动重启。

**另开一个终端**验证：

```powershell
curl http://localhost:8000/health
```

`classes_loaded` 应该等于你的类别数量。再抽查一下类别表解析对不对：

```powershell
curl "http://localhost:8000/api/v1/classes?limit=5"
```

---

## 第 4 步：先跑一张图

```powershell
python batch_run.py --input . --output ./results_test
```

（`--input .` 会扫当前目录的图片）

然后**打开 `results_test/verify/` 里的图**，确认框对齐了再往下走。

---

## 第 5 步：批量跑

```powershell
python batch_run.py --input ./images
```

耗时预估：简单图 4~5 秒，密集图受 `QWEN_MAX_TOKENS` 限制（3000 时最坏约 85 秒）。100 张图如果一成是难图，大约 25~30 分钟。

---

## 第 6 步：量化评估（有标准答案时）

如果有人工标注的标准答案（YOLO txt 格式）：

```powershell
python eval.py --gt ./labels_ground_truth --results ./results
```

输出各项指标和按类别的明细。**优先看召回率** —— 预标注场景漏检的代价远大于误检。

没有现成标准答案的话，可以用公开数据集先摸底，比如 COCO128（128 张图、80 类，自带 YOLO 标注和标准答案）：

```
https://www.kaggle.com/datasets/ultralytics/coco128
```

注意 COCO 是日常类别，跟专业领域数据差别很大，只能用来验证管道通不通、看模型的基础定位能力，**不能代表专业领域的表现**。

---

## 常见问题

**ImportError: cannot import name 'direct' from 'strategies'**

`strategies/direct.py` 不在位置上。检查目录结构：

```
项目根/
├── main.py
├── config.py
├── classes.yaml
├── core/          （__init__.py, classes.py, converter.py, parser.py, postprocess.py, qwen_client.py, visualize.py）
└── strategies/    （__init__.py, direct.py）
```

浏览器下载时可能把文件名改成带空格的（如 `direct capped.py`），检查一下。

**服务起来了但检测报 500，说类别表加载失败**

`classes.yaml` 没放对位置，或者 yaml 缩进有问题。`curl /health` 看 `classes_error` 的具体内容。

**PowerShell 里报"方法调用失败，因为 [System.String] 不包含名为 read 的方法"**

你把 Python 代码直接敲进 PowerShell 了。Python 代码要么存成 `.py` 文件跑，要么先进 `python` 交互环境。

**每次改代码都要重启服务**

用 `uvicorn main:app --reload`，改完自动重载。

**想看模型到底返回了什么**

`logs/` 目录下有每次调用的原始请求和响应（图片已脱敏成长度占位符，不占空间）。或者看 `results/raw/xxx.json` 里的 `debug.stages[0].raw_output`。

**结果目录混着不同版本代码跑的数据**

改了配置或代码后，把 `results/` 删掉重跑，否则评估结果会把新旧数据混在一起，得出错误结论。
