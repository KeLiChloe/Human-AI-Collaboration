FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py build_pool.py ./
COPY static ./static
COPY templates ./templates
COPY data/theories.json ./data/theories.json

ENV RATING_DATA_DIR=/app/data
ENV RATING_DB_PATH=/data/ratings.db
ENV HOST=0.0.0.0
ENV PORT=8765

RUN mkdir -p /data /app/data

EXPOSE 8765

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8765}"]
