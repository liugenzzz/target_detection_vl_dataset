FROM python:3.11-slim

WORKDIR /app

# fonts-wqy-zenhei 用于在验证图上正确显示中文类别名，否则会画成方块
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-wqy-zenhei && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py batch_run.py eval.py test_qwen.py draw_boxes.py ./
COPY core ./core
COPY strategies ./strategies

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
