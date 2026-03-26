# Disaster Response Coordination Environment — Dockerfile
# Builds on the OpenEnv base image which includes FastAPI, uvicorn, openenv-core

FROM ghcr.io/meta-pytorch/openenv-base:latest

LABEL maintainer="disaster-response-env"
LABEL description="Disaster Response Coordination Environment for OpenEnv"

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment source
COPY models.py ./models.py
COPY server/ ./server/

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Environment variables (override via HF Space secrets)
ENV PORT=7860
ENV HOST=0.0.0.0
ENV TASK_LEVEL=1
ENV ENABLE_WEB_INTERFACE=true

# Launch the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
