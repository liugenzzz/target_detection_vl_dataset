"""VLM 图片预标注服务。

调用自建的 Qwen3.6 全模态模型给图片打框，输出 YOLO 格式坐标。

接口：
  GET  /health                      健康检查（含类别表加载情况）
  GET  /api/v1/classes              抽查类别表解析是否正确
  POST /api/v1/detect               上传图片文件检测
  POST /api/v1/detect/base64        传 base64 检测（供其他后端服务调用）
"""

import base64
import io
import logging
import re
import time
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import config
from core import converter, postprocess, visualize
from core.classes import load_class_table
from strategies import direct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bbox-api")

app = FastAPI(title="VLM 图片预标注服务", version="1.0")


class DetectBase64Request(BaseModel):
    image_base64: str
    image_name: Optional[str] = "image.jpg"
    draw: bool = True


@app.on_event("startup")
def startup():
    """启动就加载类别表，配置错了立刻暴露，不用等第一次请求。"""
    try:
        table = load_class_table(config.CLASSES_YAML_PATH)
        logger.info("服务启动完成，类别表共 %d 个类别", table.count)
    except Exception as e:
        logger.error("类别表加载失败：%s", e)
        logger.error("服务仍会启动，但检测接口会报错。请检查 CLASSES_YAML_PATH。")


def _image_size(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return Image.open(io.BytesIO(image_bytes)).size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片：{e}")


def _run(image_bytes: bytes, mime: str, image_name: str, draw: bool):
    try:
        table = load_class_table(config.CLASSES_YAML_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"类别表加载失败：{e}")

    img_w, img_h = _image_size(image_bytes)

    start = time.time()
    try:
        result = direct.run(image_bytes, mime, table, img_w, img_h)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"调用模型服务失败：{e}")
    except Exception as e:
        logger.exception("检测执行异常")
        raise HTTPException(status_code=500, detail=f"检测执行异常：{e}")

    detections = result.get("detections", [])
    response = {
        "image": {"name": image_name, "width": img_w, "height": img_h},
        "detections": detections,
        "yolo_txt": converter.build_yolo_txt(detections),
        "stats": postprocess.build_stats(detections),
        "elapsed_sec": round(time.time() - start, 2),
        "debug": result.get("debug", {}),
    }
    if result.get("error"):
        response["error"] = result["error"]

    if draw:
        try:
            response["annotated_image_base64"] = visualize.draw_detections(
                image_bytes, detections, draw_invalid=config.DRAW_INVALID
            )
        except Exception as e:
            logger.warning("画框失败：%s", e)
            response["annotated_image_base64"] = None

    return JSONResponse(content=response)


@app.get("/health")
def health():
    info = {"status": "ok", "model": config.QWEN_MODEL, "qwen_api_url": config.QWEN_API_URL}
    try:
        table = load_class_table(config.CLASSES_YAML_PATH)
        info["classes_loaded"] = table.count
        info["classes_yaml"] = config.CLASSES_YAML_PATH
    except Exception as e:
        info["status"] = "degraded"
        info["classes_error"] = str(e)
    return info


@app.get("/api/v1/classes")
def list_classes(limit: int = 20):
    try:
        table = load_class_table(config.CLASSES_YAML_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    items = sorted(table.id2name.items())[:limit]
    return {"count": table.count, "sample": {str(k): v for k, v in items}}


@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), draw: bool = Form(True)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的文件为空")
    mime = (file.content_type or "image/jpeg").split("/")[-1].lower()
    if mime in ("jpg", "octet-stream", ""):
        mime = "jpeg"
    return _run(image_bytes, mime, file.filename or "image.jpg", draw)


@app.post("/api/v1/detect/base64")
async def detect_base64(req: DetectBase64Request):
    b64 = req.image_base64
    mime = "jpeg"
    if b64.startswith("data:"):
        header, b64 = b64.split(",", 1)
        m = re.search(r"data:image/(\w+);base64", header)
        if m:
            mime = m.group(1)
    try:
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base64 解码失败：{e}")
    return _run(image_bytes, mime, req.image_name or "image.jpg", req.draw)
