FROM python:3.11-slim

# System deps — libgomp1 for XGBoost/LightGBM OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Directories that must exist at runtime
RUN mkdir -p models db data/raw mlruns

# Non-root user for security
RUN useradd -m -u 1000 loanguard && chown -R loanguard:loanguard /app
USER loanguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
