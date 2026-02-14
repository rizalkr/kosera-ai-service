# Stage 1: Download model (Builder)
FROM python:3.11-slim as model-downloader

WORKDIR /app

# Install dependencies buat download
COPY requirements.txt .
# Install library khusus buat download model doang
RUN pip install --no-cache-dir sentence-transformers==2.3.1

# Bikin folder model & set env
RUN mkdir -p /app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Download model
COPY download_model.py .
RUN python download_model.py

# ==========================================
# Stage 2: Production Image (HF Compatible)
# ==========================================
FROM python:3.11-slim as production

WORKDIR /app

# [WAJIB HF] Buat user non-root (ID 1000)
RUN useradd -m -u 1000 user

# Install dependencies sistem (curl buat healthcheck opsional, tapi oke disimpan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [WAJIB HF] Copy model dari Stage 1 dengan ownership user 1000
# Kalau lupa --chown=user, nanti Permission Denied pas running
COPY --from=model-downloader --chown=user /app/models /app/models

# Set Env Vars
ENV TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    PYTHONUNBUFFERED=1 \
    # [WAJIB HF] Set Home ke user directory
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy code aplikasi dengan ownership user
COPY --chown=user main.py .

# [WAJIB HF] Ganti user ke non-root
USER user

# [WAJIB HF] Expose port 7860
EXPOSE 7860

# Healthcheck disesuaikan ke port 7860
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# [WAJIB HF] Jalankan di 0.0.0.0:7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]