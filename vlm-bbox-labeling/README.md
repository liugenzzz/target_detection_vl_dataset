# VLM 图片预标注服务

调用自建的 Qwen3.6 全模态模型给图片做目标预标注，输出 YOLO 格式坐标，供标注软件导入后人工复核。

本地测试见 [LOCAL_TESTING.md](LOCAL_TESTING.md)。

---

## 部署

### 1. 准备类别表

把你的类别 yaml 命名为 `classes.yaml` 放项目根目录。支持两种写法：

```yaml
nc: 347
names:
  0: "xxx"
  1: "yyy"
```

或列表写法（索引即编号）：

```yaml
names: ["xxx", "yyy"]
```

格式参考 `classes.example.yaml`。该文件通过 volume 挂载，**不会打进镜像**。

### 2. 确认网络

```bash
curl -m 10 http://192.168.78.36:3012/v1/models
```

不通先解决网络，别急着起容器。

### 3. 起服务

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

`status` 为 `ok` 且 `classes_loaded` 等于你的类别数即正常。返回 `degraded` 时看 `classes_error`。

---

## 三、配置项

改 `docker-compose.yml` 的 `environment`，重启生效。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_API_URL` | http://192.168.78.36:3012/v1/chat/completions | 后端模型地址 |
| `QWEN_API_KEY` | (已填) | 调用后端的 key |
| `QWEN_MODEL` | Qwen3.6-27B | 模型名 |
| `QWEN_TIMEOUT` | 300 | 单次请求超时（秒） |
| `QWEN_TEMPERATURE` | 0.35 | **别压到 0.1**，极低温更容易触发重复生成循环 |
| `QWEN_MAX_TOKENS` | 3000 | **难图最坏耗时的决定因素**，见下方说明 |
| `REPETITION_PENALTY` | 1.05 | 重复惩罚，别超 1.1（再高 JSON 格式会先崩） |
| `COORD_MODE` | per_mille | **坐标格式，见下方说明** |
| `ENABLE_LOGPROBS` | true | 置信度参考 |
| `MAX_BOXES_PER_CLASS` | 60 | 单类别框数上限，超出判无效。设 0 关闭 |
| `BIG_BOX_RATIO` | 0.5 | 框占整图面积超过此比例时标记可疑。设 1.0 关闭 |
| `DRAW_INVALID` | false | 验证图是否画出问题框，见下方说明 |
| `CLASSES_YAML_PATH` | /app/classes.yaml | 容器内类别表路径 |
| `SAVE_RAW_LOG` | true | 是否落盘原始请求/响应 |

### COORD_MODE（最重要）

实测 Qwen3.6 返回 0~1000 千分比坐标，默认值 `per_mille` 就是按此配置的。

可选：`per_mille` / `pixel` / `relative` / `auto`

**为什么不用 auto**：自动判断靠"坐标值有没有超出图片尺寸"来猜。图片在 480~1280 之间时，千分比坐标可能恰好都小于图片尺寸，就会被误判成像素坐标，框的位置全错。既然实测确认了格式，就不要让代码猜。

**换模型或换部署后，先用 `test_qwen.py` 重新确认坐标格式。**

### QWEN_MAX_TOKENS

这是唯一能真正拉住模型的手段。生成速度约 35 token/秒：

| 设置 | 难图最坏耗时 | 大约能容纳 |
|---|---|---|
| 3000（默认） | ~85 秒 | 60 个框 |
| 4096 | ~117 秒 | 90 个框 |
| 8192 | ~230 秒 | 180 个框 |

按你单图最多多少个目标来定，别设太大 —— 大出来的部分只会被密集场景的假框填满。截断不要紧，解析器会把已完整的框抢救出来。

### DRAW_INVALID

- `false`（默认）：验证图只画有效框，看到的就是**最终进训练集的结果**，判断标注质量用这个
- `true`：问题框也画出来（粗红框 + `[!]`），用于判断拦截是否正确

---

## 四、接口

### 健康检查
`GET /health`

### 抽查类别表
`GET /api/v1/classes?limit=5`

### 检测（上传文件）
`POST /api/v1/detect`（multipart/form-data）

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@test.jpg" -F "draw=true"
```

### 检测（base64，供其他后端调用）
`POST /api/v1/detect/base64`

```bash
curl -X POST http://localhost:8000/api/v1/detect/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<base64>","image_name":"a.jpg","draw":false}'
```

### 返回结构

```json
{
  "image": {"name": "test.jpg", "width": 640, "height": 427},
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "bbox_pixel": [250.24, 242.11, 363.52, 405.65],
      "yolo": [0.4795, 0.759, 0.177, 0.382],
      "confidence": 0.4454,
      "valid": true,
      "issues": [],
      "raw": {}
    }
  ],
  "yolo_txt": "0 0.479500 0.759000 0.177000 0.382000",
  "stats": {"total": 7, "valid": 6, "invalid": 1, "flagged_for_review": 1, "distinct_classes": 3},
  "elapsed_sec": 4.52,
  "debug": {},
  "annotated_image_base64": "..."
}
```

**关键字段：**

- `yolo_txt` —— 直接存成 `.txt` 就是标注文件，只含 `valid: true` 的框
- `valid: false` 的框仍保留在 `detections` 里，方便复查拦截是否正确
- `issues` 非空 = 需要人工重点复核。会被标记的情况：编号与名称对不上、类别不在表中、坐标越界、单框面积过大、疑似重复生成、超过单类别上限

---

## 五、批量跑 + 效果评估

```bash
pip install requests          # batch_run.py 在宿主机跑，只需要这个
python batch_run.py --input ./images
```

输出：

```
results/
├── labels/      xxx.txt          YOLO 标注文件
├── verify/      xxx_verify.jpg   验证图（人工复核看这个）
├── raw/         xxx.json         完整返回
└── summary.csv                   汇总，Excel 可打开
```

如果你的数据有人工标注的标准答案，可以量化评估：

```bash
python eval.py --gt ./labels_ground_truth --results ./results
```

输出召回率、精确率、漏检数、类别混淆，以及按类别拆开的明细 CSV。

**看指标的优先级**：预标注场景**召回率比精确率重要得多** —— 误检删一下就行，漏检要人工重新画框。

---

## 六、人工复核

看 `verify/` 目录的图，别对着坐标数字核对。

建议**按目标大小分开统计**错误率。图片分辨率在 480~1280 时，归一化宽度 0.02 的目标实际只有十几个像素，原图上本身就模糊。如果错误集中在这类极小目标，结论是"这批数据的小目标不适合 VLM 预标注"，而不是"模型能力不行" —— 这个区分影响后续决策。

---

## 七、排查

**类别表加载失败**

```bash
curl http://localhost:8000/health          # 看 classes_error
docker compose exec bbox-api ls -la /app/classes.yaml
```

**容器连不上模型服务**

```bash
docker compose exec bbox-api curl -m 10 http://192.168.78.36:3012/v1/models
```

不通就把 `docker-compose.yml` 里的 `ports` 换成 `network_mode: host`。

**框特别少或为空**

- 看 `raw/xxx.json` 里 `debug.stages[0].raw_output`，确认模型实际返回了什么
- `finish_reason` 为 `length` 说明输出被截断
- `logs/` 目录有完整的请求和响应（图片已脱敏成长度占位符）

**框的位置全错**

八成是 `COORD_MODE` 不对。用 `test_qwen.py` 重新确认模型返回的坐标格式。

---

## 八、已知边界

- **召回率只有一半左右**，人工补画的工作量要提前算进去
- **密集场景无法根治**，只能靠 `max_tokens` 限制损失
- **重复检测靠几何规律识别**（连续 5 个以上等距同尺寸框）。如果你的真实场景存在成排等距排列的同类设备（码头集装箱、货架工具），可能被误判。这类框仍完整保留在 `raw/` 里可复查
- 本服务只做转发、解析、画图，很轻量，**不需要 GPU**，算力都在后端模型那边
