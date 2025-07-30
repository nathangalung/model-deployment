FROM python:3.11-slim

# Install system dependencies for ML, H2O, and KFP
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    gcc \
    g++ \
    git \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and symlink to /usr/local/bin for global access
RUN curl -Ls https://astral.sh/uv/install.sh | bash \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && chmod +x /usr/local/bin/uv

# Set environment variables for H2O and Python compatibility
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONPATH=/app
ENV H2O_DISABLE_FLOW=true
ENV H2O_SHUTDOWN_HOOK=false
ENV H2O_LOG_LEVEL=WARN
ENV H2O_NO_PROXY=true

# Ensure proper logging for Kubeflow
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

RUN mkdir -p /app /tmp /data
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN uv pip install --system -r requirements.txt

# Test installation
RUN uv run python -c "import pandas, numpy, sklearn, h2o, mlflow, boto3, joblib, lightgbm; print('All packages installed successfully')"

CMD ["python"]