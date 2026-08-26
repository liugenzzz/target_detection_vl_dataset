"""FastAPI service for YOLO-like visual grounding SFT conversion."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from .yolo_visual_grounding_sft import convert_annotation_payload
except ImportError:  # pragma: no cover - supports direct script execution.
    from yolo_visual_grounding_sft import convert_annotation_payload


class ConvertRequest(BaseModel):
    """Request body for converting one annotation payload."""

    annotation: Dict[str, Any] = Field(..., description="YOLO-like annotation payload.")


class ApiResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


app = FastAPI(title="Grounding VQA SFT API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _parse_request_body(raw_body: bytes) -> ConvertRequest:
    text = raw_body.decode("utf-8").strip()
    if not text:
        raise ValueError("Request body is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_escape_windows_backslashes(text))
    if hasattr(ConvertRequest, "model_validate"):
        return ConvertRequest.model_validate(payload)
    return ConvertRequest.parse_obj(payload)


def _escape_windows_backslashes(text: str) -> str:
    """Make common unescaped Windows paths valid inside JSON strings."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


@app.post("/v1/grounding/convert", response_model=ApiResponse)
async def convert(request: Request) -> ApiResponse:
    try:
        convert_request = _parse_request_body(await request.body())
        result = convert_annotation_payload(
            convert_request.annotation,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApiResponse(code=0, message="success", data=result)


"""
接口输入格式:
POST /v1/grounding/convert
{
  "annotation": {
    "id": "00001",
    "image": "00100.jpg",
    "image_width": 1000,
    "image_height": 500,
    "shapes": [
      {"label": "track", "points": {"x": 169, "width": 68, "y": 181, "height": 63}}
    ]
  }
}

接口输出格式:
{
  "code": 0,
  "message": "success",
  "data": {
    "stats": {"annotations": 1, "boxes": 1, "samples": 3},
    "samples": [
      {
        "id": "00001_vlm_grounding_qa_0",
        "images": ["00100.jpg"],
        "conversations": [
          {"from": "human", "value": "<image>\\n..."},
          {"from": "gpt", "value": "[{\\"bbox_2d\\":[169,362,237,488],\\"label\\":\\"track\\"}]"}
        ],
        "metadata": {
          "source_annotation": "00001",
          "source_image": "00100.jpg",
          "image_width": 1000,
          "image_height": 500,
          "coordinate_mode": "qwen_relative_1000",
          "bbox_scale": 1000,
          "task_type": "all_objects"
        }
      }
    ]
  }
}
"""
