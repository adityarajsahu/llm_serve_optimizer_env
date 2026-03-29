# ─────────────────────────────────────────────────────────────
# LLM Deployment Optimizer — OpenEnv Environment
#
# CPU-only image — safe for HF Spaces (2 vCPU, 8 GB RAM).
# No GPU or vLLM needed; the simulator uses the lookup table.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Keeps Python output unbuffered (important for HF Space logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEPLOY_ENV_MODE=simulate

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install OS deps (curl for health-check probes)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source
COPY --chown=user . $HOME/app

# Expose the default OpenEnv port
EXPOSE 8000

# Switch to the non-root user
USER user

# Lightweight health check — HF pings this to verify the Space is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]