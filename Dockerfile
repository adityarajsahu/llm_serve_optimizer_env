# ─────────────────────────────────────────────────────────────
# LLM Deployment Optimizer — OpenEnv Environment
#
# CPU-only image — safe for HF Spaces (2 vCPU, 8 GB RAM).
# No GPU or vLLM needed; the simulator uses the lookup table.
# ─────────────────────────────────────────────────────────────

FROM vllm/vllm-openai-cpu:latest-x86_64

# Keeps Python output unbuffered (important for HF Space logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_CPU_KVCACHE_SPACE=1

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install OS deps (curl for health-check probes, git and git-lfs to clone models)
RUN apt-get update && apt-get install -y --no-install-recommends curl git git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install uv — fast Rust-based Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH" \
    UV_SYSTEM_PYTHON=1

RUN mkdir -p $HOME/app/models && cd $HOME/app/models && \
    git clone --depth 1 https://huggingface.co/EleutherAI/pythia-70m-deduped && rm -rf pythia-70m-deduped/.git && \
    git clone --depth 1 https://huggingface.co/openai-community/gpt2 && rm -rf gpt2/.git && \
    git clone --depth 1 https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct && rm -rf SmolLM2-135M-Instruct/.git

# Install Python deps first (layer-cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock /tmp/pkg/
RUN cd /tmp/pkg && uv sync --frozen --no-install-project

# Copy source
COPY --chown=user . $HOME/app

# Expose the default OpenEnv port
EXPOSE 7860

# Switch to the non-root user
USER user

# Lightweight health check — HF pings this to verify the Space is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Reset the entrypoint to bypass the base image's vllm CLI
ENTRYPOINT ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]