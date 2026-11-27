# CiteAnalyzer - Clasificación de la función de cita

Proyecto Final Despliegue de Soluciones - MAIA Uniandes - **Grupo 26**

```ini
Christian Alberto Torres Manigua # ca.torres@uniandes.edu.co
Fredy Alexander Gamez Rodriguez  # fredygamez@uniandes.edu.co
```

Cuando se escribe un artículo científico, no siempre queda claro para qué se cita un trabajo previo. 

El proyecto **CiteAnalyzer** responde esta pregunta: 

>  ***¿Cómo clasificar automáticamente el propósito de una cita usando modelos de lenguaje (NLP)?***
>
>  Para ello, asigna a cada contexto de cita una de nueve funciones: `Background`, `Gap`, `Basis`, `Comparison`, `Application`, `Improvement/Modification`, `Evidence`, `Identification of the Originator` y `Further Reading`.

Se contemplan estos componentes para crear una solución completa para el usuario final:

- **Datos:** Dataset de creación propia.
- **Modelos:** SciBERT con *fine-tuning*, Gemma3 4b, Nemotron 4.
- **Solución Web Interactiva:** API FastAPI y Tablero desplegados con contenedores Docker via Railway.

| Recurso | URL |
| --- | --- |
| Código | <https://github.com/christian1521/proyecto> |
| Tablero | <https://cite-proyecto-grupo26-production.up.railway.app/cite/> |
| API | <https://cite-proyecto-grupo26-production.up.railway.app/docs/> |
| Docker | <http://ghcr.io/christian1521/cite-proyecto-grupo26:latest> |
| Docker pull | `docker pull ghcr.io/christian1521/cite-proyecto-grupo26` |

### Inicio rápido

```bash
docker pull ghcr.io/christian1521/cite-proyecto-grupo26:latest
docker run -d -p 8001:8001 -e PORT=8001 christian1521/cite-proyecto-grupo26:latest
# http://localhost:8001/cite/
```



# Parámetros de configuración de la aplicación

Tabla basada en las variables que consume la aplicación y en los valores definidos por defecto en 

[app/config.py](../app/config.py), [app/api.py](../app/api.py), [Dockerfile](../Dockerfile) y [run.sh](../run.sh)

| Grupo                     | Parámetro                        |                                Predeterminado | Uso                                                          |
| ------------------------- | -------------------------------- | --------------------------------------------: | ------------------------------------------------------------ |
| **Obligatorio**           | `API_KEY_NVIDIA`                 |                      Valor enviado por correo | Clave para autenticarse con NVIDIA. Es obligatoria para usar ese modelo; sin ella, la aplicación puede iniciar, pero las solicitudes a NVIDIA fallan. |
| General                   | `PORT`                           |                                        `8001` | Puerto donde Uvicorn expone la aplicación.                   |
| General                   | `LOGGING_LEVEL`                  |                                 `INFO` (`20`) | Nivel de detalle de los registros.                           |
| Optimización del servicio | `PRELOAD_MODELS`                 | `false` en la configuración; `true` en Docker | Si es `true`, precarga SciBERT, SPECTER2 y Gemma 3 al iniciar. Reduce la latencia de la primera inferencia, pero el tiempo de arranque. |
| Optimización del servicio | `WEB_CONCURRENCY`                |                                 `1` en Docker | Cantidad de procesos de Uvicorn; más procesos también pueden aumentar el uso de RAM. |
| Optimización del servicio | `OMP_NUM_THREADS`                |                                 `8` en Docker | Hilos para operaciones numéricas. Ayuda a controlar el cómputo de SciBERT cuando corre en CPU, evitando que una operación use demasiados hilos y compita con el resto del servicio. |
| Optimización del servicio | `MKL_NUM_THREADS`                |                                 `8` en Docker | Hilos para operaciones de la biblioteca Intel. Ayuda a controlar el uso de CPU durante inferencias de SciBERT si PyTorch está usando MKL (Math Kernel Library) |
| Optimización del servicio | `TOKENIZERS_PARALLELISM`         |                             `false` en Docker | Desactiva el paralelismo interno de los tokenizadores de Hugging Face. Evita que los hilos de SciBERT y SPECTER2  compitan con los de PyTorch |
| **SciBERT**               | `MAX_LENGTH`                     |                                         `512` | Límite de tokens para ensamblar la entrada de SciBERT; reducirlo puede reducir el costo de inferencia, a cambio de truncar más contexto. |
| **Gemma 3**               | `GEMMA3_TIMEOUT_SECONDS`         |                                         `120` | Tiempo máximo de espera por una respuesta de Ollama.         |
| **Gemma 3**               | `GEMMA3_NUM_CTX`                 |                                        `8192` | Contexto máximo solicitado para la inferencia.               |
| **Gemma 3**               | `GEMMA3_NUM_THREAD`              |                                           `8` | Hilos de CPU por inferencia de Gemma 3.                      |
| **Gemma 3**               | `GEMMA3_NUM_PREDICT`             |                                         `128` | Máximo de tokens que se generarán.                           |
| **Gemma 3**               | `GEMMA3_NUM_BATCH`               |                                         `128` | Tamaño del lote de tokens procesado por inferencia.          |
| **Gemma 3 / Ollama**      | `OLLAMA_NUM_PARALLEL`            |                                 `1` en Docker | Inferencias que Ollama puede procesar en paralelo.           |
| **Gemma 3 / Ollama**      | `OLLAMA_FLASH_ATTENTION`         |                                 `1` en Docker | Habilita Flash Attention en Ollama si el entorno lo soporta. |
| **Gemma 3 / Ollama**      | `OLLAMA_MAX_QUEUE`               |                                 `8` en Docker | Solicitudes que pueden esperar en la cola de Ollama.         |
| **Gemma 3 / Ollama**      | `OLLAMA_KV_CACHE_TYPE`           |                              `q8_0` en Docker | Tipo de cuantización de la caché KV.                         |
| **Gemma 3 / Ollama**      | `OLLAMA_KEEP_ALIVE`              |                               `24h` en Docker | Tiempo que Ollama mantiene el modelo cargado en memoria.     |
| **NVIDIA**                | `NVIDIA_CONNECT_TIMEOUT_SECONDS` |                                          `20` | Tiempo máximo en segundos para establecer la conexión con NVIDIA. |
| **NVIDIA**                | `NVIDIA_TIMEOUT_SECONDS`         |                                         `300` | Tiempo máximo de espera de la respuesta de NVIDIA.           |



### Documentación y manuales

1. [Indice](./docs/index.md)  
2. [Manual de usuario](./docs/manual-usuario.md) 
3. [Manual de instalación](./docs/manual-instalacion.md) 
4. [Vista de desarrollo](./docs/desarrollo.md) 
5. [Retos](./docs/retos.md)



---

Enlaces: [Indice](./docs/index.md) · [Manual de usuario](./docs/manual-usuario.md) · [Manual de instalación](./docs/manual-instalacion.md) · [Vista de desarrollo](./docs/desarrollo.md) · [Retos](./docs/retos.md)

