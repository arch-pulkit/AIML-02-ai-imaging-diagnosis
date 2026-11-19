# Inference worker: polls Orthanc, runs DenseNet121 + Grad-CAM on CPU, POSTs results.
# For education only, not clinical use. Build context = repo root (see .dockerignore).
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY services/worker/requirements-torch.txt services/worker/requirements.txt /tmp/
RUN pip install --index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-torch.txt \
 && pip install -r /tmp/requirements.txt

COPY params.yaml ./params.yaml
COPY reports/thresholds.json ./reports/thresholds.json
COPY models/densenet121_best.pt ./models/densenet121_best.pt
COPY src ./src

RUN useradd --create-home --uid 1000 worker && mkdir -p /state && chown worker /state
USER worker

CMD ["python", "-m", "src.serve.worker"]