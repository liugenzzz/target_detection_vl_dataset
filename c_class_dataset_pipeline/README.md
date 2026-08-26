# 图中目标标注数据集构建

面向 **Qwen3-VL-8B-Instruct** 微调的多轮视觉定位 SFT 语料构建。

需求与边界规划文档：<https://claude.ai/code/artifact/15382193-2a60-4047-b67c-85ea8c664080>

---

## 当前状态

> **这是验证阶段的代码，不是最终实现。**

已完成的是**管道可行性验证** —— 证明「标注文件 → 0~1000 坐标 → SFT 样本」这条链路走得通，
并测出了规划文档里那些阈值和规模数字所依据的真实分布。

按规划文档的定稿，正式实现还要做三处改动：

| 项 | 现状 | 目标 |
|---|---|---|
| 样本结构 | 单轮问答 | **三轮递进**（识别 → 定位 → 描述），见文档第 04 节 |
| 项目依赖 | 通过 `sys.path` 挂载两个老项目 | **独立自持**，老项目只作参考实现 |
| 描述语句 | 规则模板 | **调自建 Qwen 服务生成**，同时产出视觉指代 |
| 质量过滤 | 无 | **三条过滤规则**，见文档第 06 节 |

已经稳定、正式实现里会保留的部分：坐标换算与钳位、YOLO 标注解析、
类别表编号↔名称双向校验、分布统计脚本。

---

## 目录

```
pipeline/
  deps.py                  两个参考项目的挂载点（正式实现里会去掉）
  yolo_gt_adapter.py       YOLO 标注 -> 标注 payload
  bbox_service_adapter.py  VLM 预标注服务 -> 标注 payload（本期不走这条路）
  describe.py              描述语句生成（当前为模板，将改为 VLM）
  schema.py                样本结构定义与字段校验
  build_dataset.py         管道编排

scripts/
  analyze_dataset.py       分布统计，用于确定过滤阈值 ★
  run_on_coco128.py        端到端联调
  estimate_scale.py        规模测算
  run_production.py        走 VLM 预标注服务（本期不用）
  run_from_batch_results.py 复用 batch_run.py 的落盘结果（本期不用）
```

---

## 用法

```bash
pip install -r requirements.txt
```

**统计分布、确定过滤阈值**（拿到正式标注数据后第一件事）：

```bash
python scripts/analyze_dataset.py                    # 默认跑 COCO128
python scripts/analyze_dataset.py --labels-dir <你的标注目录> \
    --images-dir <图片目录> --classes-yaml <类别表>
```

**端到端联调**：

```bash
python scripts/run_on_coco128.py
```

---

## 关于 COCO128

`vlm-bbox-labeling/coco128/` 下的 128 张图（80 类，自带 YOLO 标准答案）
**只用于跑通管道和测算规模，不进训练集**。

老项目的文档已经说清楚了边界：

> COCO 是日常类别，跟专业领域数据差别很大，只能用来验证管道通不通、
> 看模型的基础定位能力，**不能代表专业领域的表现**。

同样地，规划文档里的三个过滤阈值是按 COCO128 分布定的**初始值**，
正式数据到手后必须用 `analyze_dataset.py` 重跑、重定。

---

## 两个参考项目

| 项目 | 借鉴什么 |
|---|---|
| `vlm-bbox-labeling` | 类别表加载与双向校验；坐标越界裁剪思路；调自建 Qwen 服务的客户端范式 |
| `qweb3vl_grouding_vqa_lp_gai` | 像素框 → 0~1000 坐标换算与钳位；空间指代短语生成 |

两者是**参考实现**，不是运行时依赖 —— 新项目需要独立部署。
当前 `deps.py` 还在用 `sys.path` 挂载，属于验证期的临时做法，正式实现里会移除。
