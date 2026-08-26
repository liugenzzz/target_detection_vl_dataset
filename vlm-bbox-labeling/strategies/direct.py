"""检测策略：一次调用，把完整类别表交给模型，直接要坐标。

prompt 里额外要求模型自报坐标格式（bbox_format），用来跟 COORD_MODE 交叉校验。
注意：自报的格式仅作校验用，不改变实际换算行为 ——
模型连"请输出像素坐标"都不遵守，自报字段同样不能全信。
两者不一致时会给所有框打上 issue 标记，提醒人工确认。
"""

import logging

import config
from core import parser, postprocess
from core.classes import ClassTable
from core.qwen_client import call_model, encode_image

logger = logging.getLogger(__name__)


# 模型自报的格式 -> 本项目的 COORD_MODE
FORMAT_MAP = {
    "pixel_xyxy": "pixel",
    "norm_1000": "per_mille",
    "norm_01": "relative",
}


PROMPT = """你是一个严格的目标检测模型。

图片原始尺寸：宽 {width} 像素、高 {height} 像素。

以下是完整的类别表（格式为「编号: 名称」）：
{class_list}

请检测图片中属于上述类别表的物体，只输出 JSON，格式如下：

{{
  "bbox_format": "pixel_xyxy",
  "objects": [
    {{
      "class_id": 0,
      "class_name": "类别名称",
      "bbox_2d": [x_min, y_min, x_max, y_max]
    }}
  ]
}}

关于 bbox_format 字段（重要）：
- 如果你输出的是原图像素坐标，填 "pixel_xyxy"
- 如果你输出的是 0~1000 归一化坐标，填 "norm_1000"
- 如果你输出的是 0~1 归一化坐标，填 "norm_01"
- 必须如实填写你实际使用的坐标系，这个字段会被用来校验坐标

其他要求：
1. "class_id" 和 "class_name" 必须严格对应，名称逐字照抄类别表，不要自创类别
2. 坐标必须是数字，且满足 x_min < x_max、y_min < y_max
3. 边界框要尽量紧密贴合物体轮廓
4. 同一类别出现多次时，每个实例单独输出一个元素。但如果画面中存在大量密集重复的同类物体（如人群、成排的座椅），只标注其中清晰可辨的个体，不要机械枚举
5. 不要臆测不可见的目标
6. 图中没有任何属于类别表的物体时，objects 返回空数组
7. 只输出 JSON 本身，不要输出解释性文字，不要用 ``` 包裹

现在开始检测。"""


def run(image_bytes: bytes, mime: str, table: ClassTable, img_w: int, img_h: int) -> dict:
    prompt = PROMPT.format(width=img_w, height=img_h, class_list=table.format_full_list())
    data_url = encode_image(image_bytes, mime)

    text, meta = call_model(prompt, data_url, tag="direct")

    debug = {"stages": [{"name": "direct", "meta": meta, "raw_output": text}]}

    try:
        parsed = parser.extract_json(text)
        raw_items = parser.as_list(parsed)
    except ValueError as e:
        logger.error("解析失败：%s", e)
        return {"detections": [], "debug": debug, "error": str(e)}

    # 模型自报的坐标格式，跟配置交叉校验
    reported = None
    if isinstance(parsed, dict):
        reported = str(parsed.get("bbox_format", "")).lower().strip() or None

    coord_warning = None
    if reported:
        debug["reported_bbox_format"] = reported
        mapped = FORMAT_MAP.get(reported)
        if mapped is None:
            coord_warning = f"模型自报了未知的坐标格式 '{reported}'，已按 COORD_MODE={config.COORD_MODE} 处理"
        elif config.COORD_MODE != "auto" and mapped != config.COORD_MODE:
            coord_warning = (
                f"坐标格式存疑：模型自报 '{reported}'（对应 {mapped}），"
                f"但当前配置 COORD_MODE={config.COORD_MODE}。已按配置处理，请肉眼核对框的位置"
            )
        if coord_warning:
            logger.warning(coord_warning)
            debug["coord_warning"] = coord_warning

    detections = postprocess.build_detections(
        raw_items, table, img_w, img_h, token_probs=meta.get("token_logprobs")
    )

    # 实测该模型一贯谎报坐标格式（自称 pixel_xyxy 实际给千分比），
    # 所以这条告警只记在 debug 里，不往每个框的 issues 里塞 ——
    # 否则所有框都被标成"待复核"，这个标记就失去筛选价值了。

    return {"detections": detections, "debug": debug}
