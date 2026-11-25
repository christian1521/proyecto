FROM python:3.14-slim
# Un contenedor sirve ambas partes del proyecto a través de un solo uvicorn.
#   - API       (app/api.py)  -> /api/v1/...  (FastAPI + SciBERT)
#   - Dashboard (app/cite/)   -> /cite/       (app/main.py)

# Python comportamiento dentro de los contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Herramientas build, solo en caso de dependencia que no tenga una wheel precompilada 3.14
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user que ejecuta la aplicación
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/microproyecto

# 1) Dependencias primero, así que editar código en app/ no reinstala torch
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# 2) App code
COPY app ./app
COPY run.sh ./run.sh

# 3) Folders de model y de chunk cache.
#    The model (~440 MB) is NOT baked into the image: on Railway a Volume
#    is mounted here, and locally you mount it with -v.
RUN mkdir -p models/scibert/final cache/chunks && \
    chown -R api-user:api-user /opt/microproyecto

USER api-user

# Defaults; RailWay inyecta PORT
ENV MODEL_DIR=/opt/microproyecto/models/scibert/final \
    CACHE_DIR=/opt/microproyecto/cache/chunks \
    MAX_LENGTH=128 \
    PORT=8001


EXPOSE 8001

CMD ["bash", "./run.sh"]
