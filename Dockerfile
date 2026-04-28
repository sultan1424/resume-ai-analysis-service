# ─────────────────────────────────────────
# Factor 2: Dependencies — all bundled in image
# Factor 10: Dev/Prod parity — same image everywhere
# ─────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all service files
COPY app.py .
COPY extractor.py .
COPY embedder.py .
COPY retriever.py .
COPY analyzer.py .

# Factor 7: Port Binding
EXPOSE 8081

# Factor 9: Disposability — gunicorn handles SIGTERM
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--workers", "2", "--timeout", "120", "app:app"]
