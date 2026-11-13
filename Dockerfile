FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
ENTRYPOINT ["lidar-perception"]
CMD ["demo", "--output", "/artifacts/demo"]
