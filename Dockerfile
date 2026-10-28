FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/mediagent/.cache/huggingface

WORKDIR /app

# Runtime libraries used by PyTorch / sentence-transformers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin mediagent \
    && mkdir -p /app/data/chroma /home/mediagent/.cache/huggingface \
    && chown -R mediagent:mediagent /app /home/mediagent

COPY --chown=mediagent:mediagent . .

USER mediagent

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
