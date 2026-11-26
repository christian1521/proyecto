import atexit
import io
import os
import json
import logging
import re
import tarfile
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

from app import __version__
from app.config import settings

logger = logging.getLogger("citation_api")

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_CHUNKS_DIR = BASE_DIR / "cache" / "chunks"
CACHE_CHUNKS_ARCHIVE = BASE_DIR / "cache" / "chunks.tar.gz"
SPECTER2_BASE_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
SPECTER2_MAX_LENGTH = 512
GEMMA3_MODEL = "gemma3:12b"
GEMMA3_API_URL = os.environ.get("GEMMA3_API_URL", "http://localhost:11434/api/generate")
GEMMA3_TIMEOUT_SECONDS = 120
# $env:API_KEY_NVIDIA="nvapi-XXX"
# export API_KEY_NVIDIA="nvapi-XXX"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_TIMEOUT_SECONDS = int(os.environ.get("NVIDIA_TIMEOUT_SECONDS", "300"))
NVIDIA_CONNECT_TIMEOUT_SECONDS = int(
    os.environ.get("NVIDIA_CONNECT_TIMEOUT_SECONDS", "20")
)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
MAX_CONTEXT_TOKENS = 200
CHUNK_CACHE_MAX_DOCUMENTS = 20

_tokenizer_scibert = None
_model_scibert = None
_id2label_scibert: Optional[dict] = None
_tokenizer_specter2 = None
_model_specter2 = None
_device: str = "cpu"
_chunks_archive: Optional[tarfile.TarFile] = None
_chunks_archive_path: Optional[Path] = None
_chunks_archive_lock = threading.RLock()


class CitedDocumentNotFoundError(FileNotFoundError):
    """Genera error cuando un documento arXiv no está presente en el caché local."""


class ModelServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _get_scibert_model_dir() -> Path:
    configured_dir = os.environ.get("MODEL_DIR")
    if configured_dir:
        return Path(configured_dir)
    return BASE_DIR / "models" / "scibert"


def _get_cache_paths() -> Tuple[Path, Path]:
    configured_dir = os.environ.get("CACHE_DIR")
    if configured_dir:
        cache_dir = Path(configured_dir)
        return cache_dir, cache_dir.parent / CACHE_CHUNKS_ARCHIVE.name
    return CACHE_CHUNKS_DIR, CACHE_CHUNKS_ARCHIVE


def _normalize_arxiv_id(cited_paragraphs: Optional[str]) -> Optional[str]:
    if not cited_paragraphs:
        return None

    cited_id = cited_paragraphs.strip()
    if not cited_id or cited_id.lower() == "arxiv:":
        return None

    match = re.fullmatch(
        r"(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?",
        cited_id,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("El código arXiv debe tener el formato 'arXiv:0706.3639'.")

    return match.group(1)


def _close_chunk_cache_archive() -> None:
    global _chunks_archive, _chunks_archive_path

    with _chunks_archive_lock:
        if _chunks_archive is not None:
            _chunks_archive.close()
        _chunks_archive = None
        _chunks_archive_path = None


def _read_chunk_cache_file(arxiv_id: str, suffix: str) -> Optional[bytes]:
    global _chunks_archive, _chunks_archive_path

    filename = f"{arxiv_id}.{suffix}"
    cache_dir, archive_path = _get_cache_paths()
    if archive_path.is_file():
        with _chunks_archive_lock:
            if _chunks_archive is None or _chunks_archive_path != archive_path:
                if _chunks_archive is not None:
                    _chunks_archive.close()
                try:
                    _chunks_archive = tarfile.open(archive_path, mode="r:gz")
                except (OSError, tarfile.TarError) as error:
                    _chunks_archive = None
                    _chunks_archive_path = None
                    raise ValueError(
                        f"No se pudo abrir el archivo de caché '{archive_path}'."
                    ) from error
                _chunks_archive_path = archive_path

            try:
                for member in _chunks_archive.getmembers():
                    path_parts = [
                        part
                        for part in member.name.replace("\\", "/").split("/")
                        if part not in {"", "."}
                    ]
                    if (
                        member.isfile()
                        and path_parts
                        and path_parts[-1] == filename
                        and (len(path_parts) == 1 or path_parts[-2] == "chunks")
                    ):
                        member_file = _chunks_archive.extractfile(member)
                        if member_file is None:
                            break
                        with member_file:
                            return member_file.read()
            except (OSError, EOFError, tarfile.TarError) as error:
                raise ValueError(
                    f"No se pudo leer '{filename}' dentro de '{archive_path}'."
                ) from error
        return None

    chunks_path = cache_dir / filename
    if chunks_path.is_file():
        return chunks_path.read_bytes()
    return None


atexit.register(_close_chunk_cache_archive)


@lru_cache(maxsize=CHUNK_CACHE_MAX_DOCUMENTS)
def _load_cited_chunks(arxiv_id: Optional[str]) -> List[Dict[str, str]]:
    if arxiv_id is None:
        return []

    chunks_data = _read_chunk_cache_file(arxiv_id, "json")
    if chunks_data is None:
        raise CitedDocumentNotFoundError(
            f"No se encontró el documento arXiv '{arxiv_id}' en el caché local."
        )

    try:
        document = json.loads(chunks_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"El archivo de chunks para '{arxiv_id}' contiene JSON inválido."
        ) from error

    chunks = document.get("chunks") if isinstance(document, dict) else None
    if not isinstance(chunks, list):
        raise ValueError(
            f"El archivo de chunks para '{arxiv_id}' no contiene una lista 'chunks'."
        )
    expected_ranker = f"{SPECTER2_BASE_MODEL}|{SPECTER2_ADAPTER}"
    if document.get("ranker") != expected_ranker:
        raise ValueError(
            f"El documento '{arxiv_id}' no usa el ranker SPECTER2 esperado."
        )

    validated_chunks = []
    for index, chunk in enumerate(chunks):
        if (
            not isinstance(chunk, dict)
            or not isinstance(chunk.get("text"), str)
            or not isinstance(chunk.get("section"), str)
        ):
            raise ValueError(
                f"El chunk {index + 1} del documento '{arxiv_id}' debe contener "
                "campos de texto 'text' y 'section'."
            )
        if chunk["text"].strip():
            validated_chunks.append(
                {
                    "text": chunk["text"].replace("#OTHER_CITATION", "[REF]"),
                    "embedding_text": chunk["text"],
                    "section": chunk["section"],
                }
            )

    return validated_chunks


@lru_cache(maxsize=CHUNK_CACHE_MAX_DOCUMENTS)
def _load_chunk_embeddings(arxiv_id: Optional[str], chunk_count: int):
    if arxiv_id is None:
        return None

    import numpy as np

    embeddings_data = _read_chunk_cache_file(arxiv_id, "npy")
    if embeddings_data is None:
        return None

    try:
        embeddings = np.load(io.BytesIO(embeddings_data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"No se pudieron leer los embeddings SPECTER2 de '{arxiv_id}'."
        ) from error

    if (
        embeddings.ndim != 2
        or embeddings.shape != (chunk_count, 768)
        or not np.issubdtype(embeddings.dtype, np.floating)
    ):
        raise ValueError(
            f"Los embeddings SPECTER2 de '{arxiv_id}' no coinciden con sus chunks."
        )

    return embeddings.astype(np.float32, copy=False)


def _load_specter2_if_needed():
    global _tokenizer_specter2, _model_specter2, _device

    if _model_specter2 is None:
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer_specter2 = AutoTokenizer.from_pretrained(SPECTER2_BASE_MODEL)
        _model_specter2 = AutoAdapterModel.from_pretrained(SPECTER2_BASE_MODEL)
        _model_specter2.load_adapter(
            SPECTER2_ADAPTER,
            source="hf",
            set_active=True,
        )
        _model_specter2.to(_device)
        _model_specter2.eval()


def _embed_specter2_texts(texts, tokenizer, model, device):
    import torch

    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch_texts = texts[start : start + 16]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=SPECTER2_MAX_LENGTH,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batches.append(torch.nn.functional.normalize(cls_embeddings, p=2, dim=1).cpu())

    return torch.cat(batches)


def _rank_cited_chunks(
    citing_sentence,
    chunks,
    chunk_embeddings,
    tokenizer,
    model,
    device,
):
    if not chunks:
        return []

    import torch

    query_embedding = _embed_specter2_texts(
        [citing_sentence],
        tokenizer,
        model,
        device,
    )
    if chunk_embeddings is None:
        chunk_embeddings = _embed_specter2_texts(
            [chunk["embedding_text"] for chunk in chunks],
            tokenizer,
            model,
            device,
        ).numpy()

    candidate_embeddings = torch.as_tensor(
        chunk_embeddings,
        dtype=query_embedding.dtype,
    )
    candidate_embeddings = torch.nn.functional.normalize(
        candidate_embeddings,
        p=2,
        dim=1,
    )
    similarities = candidate_embeddings @ query_embedding[0]
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: similarities[index].item(),
        reverse=True,
    )[:3]

    return [
        (chunks[index], float(similarities[index].item()))
        for index in ranked_indices
    ]


def _build_scibert_inputs(context, ranked_chunks, tokenizer, device):
    import torch

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    chunk_ids = []
    for chunk, _ in ranked_chunks:
        section = chunk["section"].strip()
        text = chunk["text"].strip()
        chunk_text = f"{section}: {text}" if section and text else text or section
        chunk_ids.append(tokenizer.encode(chunk_text, add_special_tokens=False))

    context_limit = max(0, MAX_LENGTH - len(chunk_ids) - 2)
    context_ids = context_ids[: min(MAX_CONTEXT_TOKENS, context_limit)]
    remaining = max(0, MAX_LENGTH - len(context_ids) - len(chunk_ids) - 2)

    allocations = [0] * len(chunk_ids)
    order = sorted(range(len(chunk_ids)), key=lambda index: len(chunk_ids[index]))
    for position, index in enumerate(order):
        share = remaining // (len(order) - position)
        allocations[index] = min(len(chunk_ids[index]), share)
        remaining -= allocations[index]

    input_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
    token_type_ids = [0] * len(input_ids)
    for ids, allocation in zip(chunk_ids, allocations):
        chunk_segment = ids[:allocation] + [tokenizer.sep_token_id]
        input_ids.extend(chunk_segment)
        token_type_ids.extend([1] * len(chunk_segment))

    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
    }
    return {name: tensor.to(device) for name, tensor in inputs.items()}


_CATEGORY_EXCLUSIONS = {
    "Background": (
        "Do not use it when the citation specifically identifies a gap, "
        "supports a claim, compares studies, applies or modifies a method, "
        "establishes a research foundation, credits an originator, or directs "
        "readers to further reading."
    ),
    "Gap": (
        "Do not use it for general context alone; the citing authors must "
        "identify something missing, unexplored, or insufficient in prior work."
    ),
    "Basis": (
        "Do not use it for a passing mention or general context; the cited "
        "work must form an intellectual foundation for the current study."
    ),
    "Comparison": (
        "Do not use it unless the citing text actually contrasts or compares "
        "the cited work with another work, method, result, or approach."
    ),
    "Application": (
        "Do not use it when the cited method is adapted, improved, or extended; "
        "use Improvement / Modification in that case."
    ),
    "Improvement / Modification": (
        "Do not use it when a cited method is applied without modification; "
        "use Application in that case."
    ),
    "Evidence": (
        "Do not use it for context alone; the cited work must support a claim, "
        "hypothesis, procedure, or finding."
    ),
    "Identification of the Originator": (
        "Do not use it unless the text acknowledges the original source or "
        "pioneer of an idea, method, concept, or theory."
    ),
    "Further Reading": (
        "Do not use it when the citation directly supports an argument or "
        "describes work used in the current study; it must primarily direct "
        "readers to additional information."
    ),
}


def _build_gemma3_prompt(
    citing_sentence: str,
    ranked_chunks: Optional[List[tuple]] = None,
) -> str:
    categories = []
    for name, info in CitationFunctionCatalog.CITATION_FUNCTIONS.items():
        criteria = "\n".join(f"- {criterion}" for criterion in info.criteria)
        categories.append(
            f"Name: {name}\n"
            f"Description: {info.definition}\n"
            f"Criteria:\n{criteria}\n"
            f"When NOT to use it: {_CATEGORY_EXCLUSIONS[name]}"
        )

    category_names = ", ".join(CitationFunctionCatalog.list_labels())
    chunk_context = ""
    if ranked_chunks:
        formatted_chunks = [
            f"[Chunk {index}]\n"
            f"Section: {chunk['section'] or 'Unspecified'}\n"
            f"Text: {chunk['text']}"
            for index, (chunk, _) in enumerate(ranked_chunks[:3], start=1)
        ]
        chunk_context = (
            "Retrieved passages from the cited paper, ranked by relevance "
            "(context only):\n"
            f"{chr(10).join(formatted_chunks)}\n"
        )

    return (
        "You are an expert annotator of scientific literature. Your task is "
        "to decide the citation function: the rhetorical purpose for which "
        "the citing authors use the cited paper in the text.\n"
        "Categories (each with its criteria and when NOT to use it):\n"
        f"{chr(10).join(categories)}\n"
        "Rules:\n"
        "- Classify ONLY the citation marked [CITA].\n"
        "- Decide by how the citing authors use the cited paper in the text. "
        "The cited paper's title, abstract, and retrieved passages only help "
        "you understand what that paper contributes; they do not decide the "
        "category.\n"
        "- Choose exactly ONE category.\n"
        "- CRITICAL: Do not use Background as a default choice. If there is "
        "any mention of methodology, comparison, or evidence, prefer those "
        "specific classes over Background.\n"
        f"- The value of \"function\" must be exactly one of: {category_names}\n"
        "- \"confidence\" is a decimal between 0.0 and 1.0 (for example "
        "0.32), not a percentage.\n"
        "- Answer ONLY with a JSON object and no other text: "
        "{\"function\": \"<category name>\", \"confidence\": <decimal>}\n"
        f"{chunk_context}"
        f"Citing sentence: {citing_sentence}\n"
        "Which citation function does the [CITA] citation have?\n"
        "Answer with the JSON object only."
    )


def _post_json_request(request, provider: str, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ModelServiceError(
            f"{provider} respondió con HTTP {error.code}."
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ModelServiceError(
            f"No fue posible conectar con el servicio de {provider}."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelServiceError(
            f"{provider} devolvió una respuesta inválida.",
            status_code=502,
        ) from error

    if not isinstance(response_body, dict):
        raise ModelServiceError(
            f"{provider} devolvió una respuesta JSON con formato inválido.",
            status_code=502,
        )

    return response_body


def _format_model_result(
    response_text: str,
    citing_sentence: str,
    model_name: str,
) -> dict:
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ModelServiceError(
            f"{model_name} no devolvió un objeto JSON válido.",
            status_code=502,
        ) from error

    if not isinstance(result, dict):
        raise ModelServiceError(
            f"{model_name} debe devolver un objeto JSON.",
            status_code=502,
        )

    predicted_label = result.get("function")
    confidence = result.get("confidence")
    if (
        not isinstance(predicted_label, str)
        or predicted_label not in CitationFunctionCatalog.CITATION_FUNCTIONS
    ):
        raise ModelServiceError(
            f"{model_name} devolvió una categoría que no está en el catálogo.",
            status_code=502,
        )
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise ModelServiceError(
            f"{model_name} devolvió una confianza fuera del rango 0.0–1.0.",
            status_code=502,
        )

    citation_info = CitationFunctionCatalog.get(predicted_label)
    return {
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "citing_sentence": citing_sentence,
        "all_probabilities": {},
        "definition": citation_info.definition,
        "criteria": citation_info.criteria,
        "model_name": model_name,
    }


def _classify_with_gemma3(
    citing_sentence: str,
    ranked_chunks: Optional[List[tuple]] = None,
) -> dict:
    request_body = json.dumps(
        {
            "model": GEMMA3_MODEL,
            "prompt": _build_gemma3_prompt(citing_sentence, ranked_chunks),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GEMMA3_API_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ollama_response = _post_json_request(
        request,
        provider="Ollama",
        timeout=GEMMA3_TIMEOUT_SECONDS,
    )
    response_text = ollama_response.get("response")
    if not isinstance(response_text, str):
        raise ModelServiceError(
            "La respuesta de Ollama no contiene el resultado de Gemma 3.",
            status_code=502,
        )

    return _format_model_result(response_text, citing_sentence, "Gemma 3")


def _classify_with_nvidia(
    citing_sentence: str,
    ranked_chunks: Optional[List[tuple]] = None,
) -> dict:
    api_key = settings.api_key_nvidia
    if api_key is None or not api_key.get_secret_value().strip():
        raise ModelServiceError(
            "Configure la variable de entorno API_KEY_NVIDIA para usar NVIDIA."
        )

    request_body = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": _build_gemma3_prompt(citing_sentence, ranked_chunks),
            }
        ],
        "max_tokens": 16384,
        "seed": 0,
        "stream": True,
        "reasoning_effort": "low",
    }

    try:
        with requests.post(
            NVIDIA_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept": "text/event-stream",
            },
            json=request_body,
            stream=True,
            timeout=(NVIDIA_CONNECT_TIMEOUT_SECONDS, NVIDIA_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            response_text = _read_nvidia_stream(response)
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else 502
        raise ModelServiceError(
            f"NVIDIA API respondió con HTTP {status_code}; verifica la API key y el modelo.",
            status_code=502,
        ) from error
    except requests.ConnectTimeout as error:
        raise ModelServiceError(
            "No se pudo conectar con NVIDIA antes del timeout de conexión.",
            status_code=503,
        ) from error
    except requests.ReadTimeout as error:
        raise ModelServiceError(
            "NVIDIA tardó demasiado en completar la respuesta.",
            status_code=504,
        ) from error
    except requests.Timeout as error:
        raise ModelServiceError(
            "La solicitud a NVIDIA excedió el tiempo de espera.",
            status_code=504,
        ) from error
    except requests.ConnectionError as error:
        raise ModelServiceError(
            "No fue posible establecer conexión con NVIDIA API.",
            status_code=503,
        ) from error
    except requests.RequestException as error:
        raise ModelServiceError(
            "Falló la solicitud HTTP a NVIDIA API.",
            status_code=502,
        ) from error

    return _format_model_result(
        response_text,
        citing_sentence,
        "Nemotron 3 Ultra 550B (NVIDIA)",
    )


def _read_nvidia_stream(response) -> str:
    content_parts = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ModelServiceError(
                    "NVIDIA API envió un evento SSE inválido.",
                    status_code=502,
                ) from error

        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break

        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ModelServiceError(
                "NVIDIA API envió un evento SSE inválido.",
                status_code=502,
            ) from error

        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])

    response_text = "".join(content_parts)
    if not response_text:
        raise ModelServiceError(
            "NVIDIA API finalizó el flujo sin devolver contenido.",
            status_code=502,
        )

    return response_text


def _load_model_if_needed():
    global _tokenizer_scibert, _model_scibert, _id2label_scibert, _device

    if _model_scibert is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_dir = _get_scibert_model_dir()
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"No se encontró 'config.json' en {model_dir}.")

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Cargando modelo desde: {model_dir}")

        _tokenizer_scibert = AutoTokenizer.from_pretrained(str(model_dir))
        _model_scibert = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        _model_scibert.to(_device)
        _model_scibert.eval()

        label_mapping_path = model_dir / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, "r", encoding="utf-8") as f:
                _id2label_scibert = {int(k): v for k, v in json.load(f)["id2label"].items()}
        else:
            _id2label_scibert = {int(k): v for k, v in _model_scibert.config.id2label.items()}

        logger.info(f"Modelo SCIBERT cargado en dispositivo: {_device}")
        logger.info(f"Etiquetas: {list(_id2label_scibert.values())}")


class CitationFunctionInfo(BaseModel):
    definition: str
    criteria: List[str]

class CitationFunctionCatalog:

    CITATION_FUNCTIONS: Dict[str, CitationFunctionInfo] = {
        "Background": CitationFunctionInfo(
            definition=(
                "Citations used to provide context, summarize the general "
                "background of a research topic, or trace the history of a "
                "field or idea. These citations lay the foundation for "
                "understanding the focal study."
            ),
            criteria=[
                "Reviews existing literature to ensure comprehensiveness.",
                "Cites review papers or relevant previous studies to depict "
                "the current state of the field.",
                "Highlights important or prevailing findings in the field.",
                "Traces the historical development or presents key ideas in "
                "the subject area.",
            ],
        ),
        "Gap": CitationFunctionInfo(
            definition=(
                "Citations that help identify research gaps or unexplored "
                "areas, justifying the author's choice of research topic."
            ),
            criteria=[
                "Highlights what has or has not been done in the research area.",
                "Identifies gaps in knowledge or areas for further study.",
                "Justifies the relevance and importance of the current "
                "research by contrasting it with existing work.",
            ],
        ),
        "Basis": CitationFunctionInfo(
            definition=(
                "Citations that provide a foundation for the current "
                "research. This can operate at a macro level (broad "
                "influence) or a micro level (specific key contributions)."
            ),
            criteria=[
                "Serves as the intellectual starting point for the focal research.",
                "The cited work significantly shapes the research idea or hypothesis.",
                "The focal research builds upon, continues, or expands on "
                "the cited work.",
            ],
        ),
        "Comparison": CitationFunctionInfo(
            definition=(
                "Citations used to compare the current work with cited "
                "studies or to draw comparisons between cited studies."
            ),
            criteria=[
                "Highlights similarities or differences between the current "
                "research and cited works.",
                "Compares methodologies, findings, algorithms, data, or "
                "theoretical concepts.",
                "May point out advantages of the current study over previous "
                "ones or establish links among different cited works.",
            ],
        ),
        "Application": CitationFunctionInfo(
            definition=(
                "Citations that directly employ a method, technique, or "
                "tool from the cited work without modification."
            ),
            criteria=[
                "Utilizes existing methods, algorithms, instruments, or data "
                "for the current research.",
                "Employs the cited work as a practical tool (e.g., "
                "equations, analysis methods) for calculation or "
                "experimentation.",
                "Applies the cited method without modification.",
            ],
        ),
        "Improvement / Modification": CitationFunctionInfo(
            definition=(
                "Citations in which methods or tools from the cited work "
                "are adapted, expanded, or modified for the current research."
            ),
            criteria=[
                "The cited method is adapted, improved, or extended to fit "
                "new experimental conditions or research objectives.",
                "Uses the cited work as a foundation but modifies it to "
                "enhance precision or scope.",
            ],
        ),
        "Evidence": CitationFunctionInfo(
            definition=(
                "Citations used to support claims, hypotheses, or findings "
                "in the current research."
            ),
            criteria=[
                "Supports arguments, hypotheses, or factual statements.",
                "Justifies research design, methodologies, or experimental "
                "procedures.",
                "Substantiates or explains findings, especially in the "
                "discussion section.",
                "Provides evidence to mitigate limitations or support "
                "further research suggestions.",
            ],
        ),
        "Identification of the Originator": CitationFunctionInfo(
            definition=(
                "Citations used to acknowledge the original source of an "
                "idea, concept, method, or theory."
            ),
            criteria=[
                "Identifies the original publication where a key idea or "
                "method was first introduced.",
                "Acknowledges pioneers in the field.",
                "Gives credit to the priority of a cited work, demonstrating "
                "intellectual indebtedness.",
            ],
        ),
        "Further Reading": CitationFunctionInfo(
            definition=(
                "Citations that direct readers to additional or "
                "supplementary literature for more detailed information or "
                "context."
            ),
            criteria=[
                "Alerts readers to new, different, or relevant sources of "
                "information.",
                "Provides more complete details on data, methods, or "
                "background not fully covered in the current work.",
                "Often used to refer readers to external sources for deeper "
                "insights.",
            ],
        ),
    }

    @classmethod
    def get(cls, label: str) -> Optional[CitationFunctionInfo]:
        return cls.CITATION_FUNCTIONS.get(label)

    @classmethod
    def list_labels(cls) -> List[str]:
        return list(cls.CITATION_FUNCTIONS.keys())

def predict(
    citing_sentence: str,
    type_model: Optional[str] = None,
    cited_paragraphs: Optional[str] = None,
    use_chunks: bool = True,
) -> dict:
    request_started = time.perf_counter()

    arxiv_id = _normalize_arxiv_id(cited_paragraphs)
    if use_chunks and arxiv_id and "[CITA]" not in citing_sentence:
        raise ValueError(
            "El contexto de la cita debe contener la etiqueta '[CITA]' "
            "cuando se proporciona el párrafo del documento citado."
        )

    if type_model not in {"scibert", "gemma3", "nvidia_nemotron_3_ultra_550b_a55b"}:
        raise ValueError(
            "Debe especificar un tipo de modelo válido: 'scibert', 'gemma3' "
            "o 'nvidia_nemotron_3_ultra_550b_a55b'."
        )

    chunks_cache_state = "skipped"
    chunks_cache_seconds = 0.0
    cited_chunks = []
    if use_chunks:
        cache_before = _load_cited_chunks.cache_info()
        cache_started = time.perf_counter()
        cited_chunks = _load_cited_chunks(arxiv_id)
        chunks_cache_seconds = time.perf_counter() - cache_started
        cache_after = _load_cited_chunks.cache_info()
        chunks_cache_state = (
            "hit" if cache_after.hits > cache_before.hits else "miss"
        )

    text_to_classify = (
        citing_sentence.replace("[CITA]", "#CITATION_TAG")
        if type_model == "scibert"
        else citing_sentence
    )

    scibert_model_cache = "skipped"
    scibert_model_load_seconds = 0.0
    if type_model == "scibert":
        scibert_model_cache = (
            "hit" if _model_scibert is not None else "miss"
        )
        model_load_started = time.perf_counter()
        _load_model_if_needed()
        scibert_model_load_seconds = time.perf_counter() - model_load_started
        _tokenizer = _tokenizer_scibert
        _model = _model_scibert
        _id2label = _id2label_scibert

    ranked_chunks = []
    embeddings_cache_state = "skipped"
    embeddings_cache_seconds = 0.0
    embeddings_source = "skipped"
    specter2_model_cache = "skipped"
    specter2_model_load_seconds = 0.0
    ranking_seconds = 0.0
    if cited_chunks:
        cache_before = _load_chunk_embeddings.cache_info()
        cache_started = time.perf_counter()
        chunk_embeddings = _load_chunk_embeddings(arxiv_id, len(cited_chunks))
        embeddings_cache_seconds = time.perf_counter() - cache_started
        cache_after = _load_chunk_embeddings.cache_info()
        embeddings_cache_state = (
            "hit" if cache_after.hits > cache_before.hits else "miss"
        )
        embeddings_source = (
            "precomputed_npy"
            if chunk_embeddings is not None
            else "recompute_chunk_embeddings"
        )

        specter2_model_cache = (
            "hit" if _model_specter2 is not None else "miss"
        )
        model_load_started = time.perf_counter()
        _load_specter2_if_needed()
        specter2_model_load_seconds = time.perf_counter() - model_load_started

    models_ready_at = time.perf_counter()

    if cited_chunks:
        ranking_started = time.perf_counter()
        ranked_chunks = _rank_cited_chunks(
            citing_sentence,
            cited_chunks,
            chunk_embeddings,
            _tokenizer_specter2,
            _model_specter2,
            _device,
        )
        ranking_seconds = time.perf_counter() - ranking_started

    chunks_result = [
        {
            f"chunk_text{index}": chunk["text"],
            f"chunk_score{index}": score,
            f"chunk_section{index}": chunk["section"],
        }
        for index, (chunk, score) in enumerate(ranked_chunks, start=1)
    ]

    if type_model == "gemma3":
        result = _classify_with_gemma3(citing_sentence, ranked_chunks)
        result["chunks"] = chunks_result
        return result

    if type_model == "nvidia_nemotron_3_ultra_550b_a55b":
        result = _classify_with_nvidia(citing_sentence, ranked_chunks)
        result["chunks"] = chunks_result
        return result

    import torch

    input_started = time.perf_counter()
    inputs = _build_scibert_inputs(
        text_to_classify,
        ranked_chunks,
        _tokenizer,
        _device,
    )
    input_seconds = time.perf_counter() - input_started

    inference_started = time.perf_counter()
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    inference_seconds = time.perf_counter() - inference_started

    pred_id = int(torch.argmax(probs).item())
    predicted_label = _id2label[pred_id]

    result = {
        "predicted_label": predicted_label,
        "confidence": float(probs[pred_id].item()),
        "citing_sentence": citing_sentence,
        "all_probabilities": {
            _id2label[i]: float(probs[i].item()) for i in range(len(_id2label))
        },
        "chunks": chunks_result,
    }

    citation_info = CitationFunctionCatalog.get(predicted_label)
    if citation_info is not None:
        result["definition"] = citation_info.definition
        result["criteria"] = citation_info.criteria

    logger.info(
        "prediction_timing model=%s device=%s chunks=%d "
        "chunks_cache=%s:%.3fs embeddings_cache=%s:%.3fs "
        "embedding_source=%s scibert_model_cache=%s:%.3fs "
        "specter2_model_cache=%s:%.3fs ranking=%.3fs "
        "scibert_input=%.3fs scibert_inference=%.3fs "
        "after_models_ready=%.3fs total=%.3fs",
        type_model,
        _device,
        len(cited_chunks),
        chunks_cache_state,
        chunks_cache_seconds,
        embeddings_cache_state,
        embeddings_cache_seconds,
        embeddings_source,
        scibert_model_cache,
        scibert_model_load_seconds,
        specter2_model_cache,
        specter2_model_load_seconds,
        ranking_seconds,
        input_seconds,
        inference_seconds,
        time.perf_counter() - models_ready_at,
        time.perf_counter() - request_started,
    )

    return result


class CitationRequest(BaseModel):
    citing_sentence: str
    type_model: Optional[str] = None
    cited_paragraphs: Optional[str] = None
    use_chunks: bool = True


api_router = APIRouter()


@api_router.get("/health", status_code=200)
def health():
    try:
        _load_model_if_needed()
    except Exception as e:
        return {"status": "degraded", "device": None, "model_version": __version__, "error": str(e)}

    return {
        "status": "ok",
        "device": _device,
        "model_version": __version__,
    }


@api_router.get("/citation-functions", status_code=200)
def list_citation_functions(function_name: Optional[str] = None):
    if function_name:
        info = CitationFunctionCatalog.get(function_name)
        if info is None:
            raise ValueError(
                f"No se encontró la función de cita '{function_name}'. "
                f"Funciones disponibles: {CitationFunctionCatalog.list_labels()}"
            )
        return {function_name: info.model_dump()}

    return {
        label: info.model_dump()
        for label, info in CitationFunctionCatalog.CITATION_FUNCTIONS.items()
    }

@api_router.post("/classify-citation", status_code=200)
def classify_citation(payload: CitationRequest):
    try:
        result = predict(
            payload.citing_sentence,
            type_model=payload.type_model,
            cited_paragraphs=payload.cited_paragraphs,
            use_chunks=payload.use_chunks,
        )
    except CitedDocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result