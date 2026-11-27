# CiteAnalyzer: clasificación de la función de cita

**Proyecto Final - MAIA Uniandes **

## Grupo 26

```
Christian Alberto Torres Manigua 
Fredy Alexander Gamez Rodriguez
```

**CiteAnalyzer** clasifica automáticamente el propósito discursivo de una cita académica en una de 
**9 categorías**: `Background`, `Gap`, `Basis`, `Comparison`, `Application`, `Improvement/Modification`,
`Evidence`, `Identification of the Originator` y `Further Reading`. Uso de modelos *encoder* (SciBERT) con *fine-tuning*. Se despliega con un API FastAPI y un tablero web, en un mismo contenedor Docker desplegado via Railway.

## Accesos al prototipo desplegado

| Recurso | URL |
| --- | --- |
| Repositorio código en Github | <https://github.com/christian1521/proyecto> <br>[Ver detalles](desarrollo.md) |
| Página de inicio | <https://cite-proyecto-grupo26-production.up.railway.app/> |
| API | <https://cite-proyecto-grupo26-production.up.railway.app/docs/> |
| Tablero (aplicación) | <https://cite-proyecto-grupo26-production.up.railway.app/cite/> |
| Imagen Docker | <http://ghcr.io/christian1521/cite-proyecto-grupo26:latest> <br> `christian1521/cite-proyecto-grupo26` |

## Manuales

- [Manual de instalación](manual-instalacion.md): instalación con Docker, desde código fuente y despliegue en Railway.
- [Manual de usuario](manual-usuario.md): cómo analizar una cita e interpretar el resultado.

## Documentación complementaria

- [Vista de desarrollo](desarrollo.md): estructura del repositorio, stack tecnológico, componentes y
  flujo de uso.
- [Retos y evidencias](retos.md): bitácora de pruebas de modelos, construcción de la imagen Docker y
  despliegue en Railway.

## Stack tecnológico

| Categoría | Tecnología |
| --- | --- |
| Modelos | SciBERT (*fine-tuning*) , Gemma3_4b (Local), Nvidia Nemotron 3 (API) |
| Experimentos | MLflow en AWS EC2 |
| Datos y modelos versionados | DVC con remoto AWS-S3 (`s3://dvcproject-ia/citeanalyzer`) |
| API | FastAPI + Uvicorn (`/api/v1`) |
| Tablero | HTML/JS por FastAPI (`/cite/`) |
| Contenedor | Docker (`python:3.14-slim`), una imagen para Modelos, API y Tablero |
| Registro de imágenes | Github Containers  (http://ghcr.io/christian1521/cite-proyecto-grupo26:latest) |
| Despliegue (PaaS) | Railway en contenedores Docker |
| Código fuente | Repositorio GitHub https://github.com/christian1521/proyecto |
