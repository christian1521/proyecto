flowchart TD
    U((Usuario)) -->|HTTPS| R

    subgraph R[Railway - servicio cite-api]
        subgraph C[Contenedor /christian1521/cite-proyecto-grupo26:latest]
            UV[Uvicorn <br> FastAPI app.main]
            T[Tablero /cite/]
            A[API /api/v1]
            M1[(Modelo SciBERT <br>fine-tuned)]
            M2[(Modelo Gemma3 <br>Local OpenWeitght)]
            M3[(Modelo Nemotron <br>API)]
            UV --> T
            UV --> A
            A --> M1
            A --> M2
            A --> M3
            T -. fetch JSON .-> A
        end
    end

    DH[DockerHub<br/>fredygamez/cite-api] -->|pull imagen| R
    GH[Repositorio GitHub<br/>christian1521/microproyecto] -->|docker build + push| DH
    S3[(Dataset + DVC<br/>AWS-S3<br/>christian1521-dvcstore)] -->|dvc pull modelos| GH
    ML[MLOps con MLFlow <br> AWS EC2] -.registro de experimentos.- GH