import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import __version__

logger = logging.getLogger("citation_api")

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR_SCIBERT = BASE_DIR / "models" / "scibert" / "final"
CACHE_CHUNKS_DIR = BASE_DIR / "cache" / "chunks"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

_tokenizer_scibert = None
_model_scibert = None
_id2label_scibert: Optional[dict] = None
_device: str = "cpu"


class CitedDocumentNotFoundError(FileNotFoundError):
    """Genera error cuando un documento arXiv no está presente en el caché local."""


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


def _load_cited_chunks(arxiv_id: Optional[str]) -> List[Dict[str, str]]:
    if arxiv_id is None:
        return []

    chunks_path = CACHE_CHUNKS_DIR / f"{arxiv_id}.json"
    if not chunks_path.is_file():
        raise CitedDocumentNotFoundError(
            f"No se encontró el documento arXiv '{arxiv_id}' en el caché local."
        )

    try:
        with chunks_path.open("r", encoding="utf-8") as chunks_file:
            document = json.load(chunks_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"El archivo de chunks para '{arxiv_id}' contiene JSON inválido."
        ) from error

    chunks = document.get("chunks") if isinstance(document, dict) else None
    if not isinstance(chunks, list):
        raise ValueError(
            f"El archivo de chunks para '{arxiv_id}' no contiene una lista 'chunks'."
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
                {"text": chunk["text"].replace('#OTHER_CITATION', '[REF]'), "section": chunk["section"]}
            )

    return validated_chunks


def _embed_texts(texts, tokenizer, model, device):
    import torch

    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch_texts = texts[start : start + 16]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            outputs = model.base_model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            pooled_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            pooled_embeddings = pooled_embeddings / attention_mask.sum(dim=1).clamp(min=1)
            batches.append(
                torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1).cpu()
            )

    return torch.cat(batches)


def _rank_cited_chunks(citing_sentence, chunks, tokenizer, model, device):
    if not chunks:
        return []

    chunk_texts = [
        f"{chunk['section']} {chunk['text']}".strip()
        for chunk in chunks
    ]
    embeddings = _embed_texts(
        [citing_sentence] + chunk_texts,
        tokenizer,
        model,
        device,
    )
    similarities = embeddings[1:] @ embeddings[0]
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: similarities[index].item(),
        reverse=True,
    )[:3]

    return [
        (chunks[index], float(similarities[index].item()))
        for index in ranked_indices
    ]


def _load_model_if_needed():
    global _tokenizer_scibert, _model_scibert, _id2label_scibert, _device

    if _model_scibert is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not (MODEL_DIR_SCIBERT / "config.json").exists():
            raise FileNotFoundError(f"No se encontró 'config.json' en {MODEL_DIR_SCIBERT}.")

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Cargando modelo desde: {MODEL_DIR_SCIBERT}")

        _tokenizer_scibert = AutoTokenizer.from_pretrained(str(MODEL_DIR_SCIBERT))
        _model_scibert = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR_SCIBERT))
        _model_scibert.to(_device)
        _model_scibert.eval()

        label_mapping_path = MODEL_DIR_SCIBERT / "label_mapping.json"
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

def predict(citing_sentence: str, type_model: Optional[str] = None, cited_paragraphs: Optional[str] = None) -> dict:

    arxiv_id = _normalize_arxiv_id(cited_paragraphs)
    if arxiv_id and "[CITA]" not in citing_sentence:
        raise ValueError(
            "El contexto de la cita debe contener la etiqueta '[CITA]' "
            "cuando se proporciona el párrafo del documento citado."
        )

    if type_model != "scibert":
        raise ValueError(
            "Debe especificar el tipo de modelo a utilizar: 'scibert'."
        )

    cited_chunks = _load_cited_chunks(arxiv_id)
    _load_model_if_needed()

    import torch

    text_to_classify = citing_sentence.replace("[CITA]", "<citation>")

    _tokenizer = _tokenizer_scibert
    _model = _model_scibert
    _id2label = _id2label_scibert

    ranked_chunks = _rank_cited_chunks(
        text_to_classify,
        cited_chunks,
        _tokenizer,
        _model,
        _device,
    )
    chunk_references = [
        f"{chunk['section']} {chunk['text']}".strip()
        for chunk, _ in ranked_chunks
    ]
    text_reference = (
        f" {_tokenizer.sep_token} ".join(chunk_references)
        if chunk_references
        else None
    )

    inputs = _tokenizer(
        text_to_classify,
        text_reference,
        truncation="only_second",
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    predicted_label = _id2label[pred_id]

    result = {
        "predicted_label": predicted_label,
        "confidence": float(probs[pred_id].item()),
        "citing_sentence": citing_sentence,
        "all_probabilities": {
            _id2label[i]: float(probs[i].item()) for i in range(len(_id2label))
        },
        "chunks": [
            {
                f"chunk_text{index}": chunk["text"],
                f"chunk_score{index}": score,
                f"chunk_section{index}": chunk["section"],
            }
            for index, (chunk, score) in enumerate(ranked_chunks, start=1)
        ],
    }

    citation_info = CitationFunctionCatalog.get(predicted_label)
    if citation_info is not None:
        result["definition"] = citation_info.definition
        result["criteria"] = citation_info.criteria

    return result


class CitationRequest(BaseModel):
    citing_sentence: str
    type_model: Optional[str] = None
    cited_paragraphs: Optional[str] = None


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
    """
    Devuelve el catálogo de funciones de cita soportadas, con su Definition y Criteria.
    """
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
async def classify_citation(payload: CitationRequest):
    try:
        result = predict(payload.citing_sentence, type_model=payload.type_model, cited_paragraphs=payload.cited_paragraphs)
    except CitedDocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result