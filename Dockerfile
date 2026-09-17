FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

# Railway injects $PORT at runtime; default to 8000 for local use
ENV PORT=8000
EXPOSE 8000

# Fix #8: explicit uvicorn with $PORT so Railway's assigned port is always honoured
CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port ${PORT}"]
