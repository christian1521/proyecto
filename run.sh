#!/usr/bin/env bash
# Arranque unico: Ollama (interno, 127.0.0.1:11434) + uvicorn (API /api/v1 y tablero /cite).
set -e
# 1) Ollama en segundo plano (usa OLLAMA_HOST y OLLAMA_MODELS del Dockerfile)
ollama serve &
# 2) Esperar a que Ollama responda (max ~30 s, sondeo cada 0.5 s)
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 0.5
done
if ollama list | grep -q "${OLLAMA_MODEL%%:*}"; then
  echo "Ollama listo (${OLLAMA_MODEL})"
  # 3) Precargar el modelo en RAM sin bloquear el arranque
  curl -sf http://127.0.0.1:11434/api/generate \
       -d "{\"model\":\"${OLLAMA_MODEL}\",\"keep_alive\":\"${OLLAMA_KEEP_ALIVE:-24h}\"}" >/dev/null \
       && echo "Modelo ${OLLAMA_MODEL} cargado en memoria" &
else
  echo "AVISO: modelo ${OLLAMA_MODEL} no encontrado en ${OLLAMA_MODELS}"
fi
# 4) uvicorn al frente; PORT lo inyecta Railway, 8001 es el default local.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
