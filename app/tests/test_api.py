from pathlib import Path
 
import pandas as pd
import pytest
 
from app import api as api_module
from app.api import predict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR_SCIBERT_AVAILABLE = BASE_DIR / "models" / "scibert" / "final"
CACHED_DOCUMENT_AVAILABLE = BASE_DIR / "cache" / "chunks" / "0706.3639.json"

@pytest.mark.skipif(
    not MODEL_DIR_SCIBERT_AVAILABLE.exists() or not CACHED_DOCUMENT_AVAILABLE.exists(),
    reason="No se encontró el modelo SciBERT o el documento de prueba en caché.",
)
def test_make_prediction(test_data: pd.DataFrame) -> None:
    # Given
    sample = test_data.iloc[0]
 
    # When
    prediction_data = predict(
        citing_sentence=sample["citing_sentence"],
        type_model="scibert",
        cited_paragraphs="arXiv:0706.3639",
    )
 
    # Then
    assert prediction_data["predicted_label"]
    assert 0.0 <= prediction_data["confidence"] <= 1.0
    assert len(prediction_data["chunks"]) == 3
    assert all(
        {"chunk_text" + str(index), "chunk_score" + str(index), "chunk_section" + str(index)}
        <= chunk.keys()
        for index, chunk in enumerate(prediction_data["chunks"], start=1)
    )
    assert [
        chunk[f"chunk_score{index}"]
        for index, chunk in enumerate(prediction_data["chunks"], start=1)
    ] == sorted(
        (
            chunk[f"chunk_score{index}"]
            for index, chunk in enumerate(prediction_data["chunks"], start=1)
        ),
        reverse=True,
    )


def test_bert_model_is_not_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_load_model_if_needed",
        lambda: pytest.fail("No debe cargarse un modelo para una selección inválida."),
    )

    with pytest.raises(ValueError, match="'scibert'"):
        predict(citing_sentence="A citation sentence.", type_model="bert")


def test_invalid_arxiv_id_is_rejected_before_loading_model(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_load_model_if_needed",
        lambda: pytest.fail("No debe cargarse un modelo con un código arXiv inválido."),
    )

    with pytest.raises(ValueError, match="código arXiv"):
        predict(
            citing_sentence="A citation [CITA].",
            type_model="scibert",
            cited_paragraphs="../0706.3639",
        )