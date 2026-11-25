from pathlib import Path
import io
import json
import tarfile
 
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import SecretStr
 
from app import api as api_module
from app.api import predict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR_SCIBERT_AVAILABLE = BASE_DIR / "models" / "scibert" / "final"
CACHED_DOCUMENT_AVAILABLE = BASE_DIR / "cache" / "chunks" / "0706.3639.json"
CACHED_EMBEDDINGS_AVAILABLE = BASE_DIR / "cache" / "chunks" / "0706.3639.npy"

@pytest.mark.skipif(
    not MODEL_DIR_SCIBERT_AVAILABLE.exists()
    or not CACHED_DOCUMENT_AVAILABLE.exists()
    or not CACHED_EMBEDDINGS_AVAILABLE.exists(),
    reason="No se encontró SciBERT o los archivos de prueba en caché.",
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

    api_module._load_specter2_if_needed()
    cached_chunks = api_module._load_cited_chunks("0706.3639")
    candidate_embedding = api_module._embed_specter2_texts(
        [cached_chunks[1]["embedding_text"]],
        api_module._tokenizer_specter2,
        api_module._model_specter2,
        api_module._device,
    )[0].numpy()
    cached_embedding = api_module._load_chunk_embeddings(
        "0706.3639",
        len(cached_chunks),
    )[1]
    similarity = np.dot(candidate_embedding, cached_embedding) / (
        np.linalg.norm(candidate_embedding) * np.linalg.norm(cached_embedding)
    )
    assert similarity > 0.99999


def test_chunk_cache_can_be_read_from_tar_gz(monkeypatch, tmp_path) -> None:
    arxiv_id = "9999.99999"
    archive_path = tmp_path / "cache" / "chunks.tar.gz"
    archive_path.parent.mkdir()
    document = {
        "ranker": "allenai/specter2_base|allenai/specter2",
        "chunks": [{"text": "Cached paragraph.", "section": "Results"}],
    }
    embedding = np.ones((1, 768), dtype=np.float32)
    embedding_buffer = io.BytesIO()
    np.save(embedding_buffer, embedding)

    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, contents in (
            (f"chunks/{arxiv_id}.json", json.dumps(document).encode("utf-8")),
            (f"chunks/{arxiv_id}.npy", embedding_buffer.getvalue()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))

    monkeypatch.setattr(api_module, "CACHE_CHUNKS_ARCHIVE", archive_path)
    monkeypatch.setattr(api_module, "CACHE_CHUNKS_DIR", tmp_path / "unpacked")

    try:
        chunks = api_module._load_cited_chunks(arxiv_id)
        loaded_embeddings = api_module._load_chunk_embeddings(arxiv_id, len(chunks))
    finally:
        api_module._close_chunk_cache_archive()

    assert chunks == [
        {
            "text": "Cached paragraph.",
            "embedding_text": "Cached paragraph.",
            "section": "Results",
        }
    ]
    np.testing.assert_array_equal(loaded_embeddings, embedding)


def test_chunk_data_is_cached_in_memory(monkeypatch) -> None:
    arxiv_id = "9998.88888"
    document = {
        "ranker": "allenai/specter2_base|allenai/specter2",
        "chunks": [{"text": "Cached paragraph.", "section": "Results"}],
    }
    embedding = np.ones((1, 768), dtype=np.float32)
    embedding_buffer = io.BytesIO()
    np.save(embedding_buffer, embedding)
    file_contents = {
        "json": json.dumps(document).encode("utf-8"),
        "npy": embedding_buffer.getvalue(),
    }
    read_files = []

    def fake_read_chunk_cache_file(requested_id, suffix):
        read_files.append((requested_id, suffix))
        return file_contents[suffix]

    monkeypatch.setattr(
        api_module,
        "_read_chunk_cache_file",
        fake_read_chunk_cache_file,
    )
    api_module._load_cited_chunks.cache_clear()
    api_module._load_chunk_embeddings.cache_clear()

    try:
        chunks_first = api_module._load_cited_chunks(arxiv_id)
        chunks_second = api_module._load_cited_chunks(arxiv_id)
        embeddings_first = api_module._load_chunk_embeddings(arxiv_id, 1)
        embeddings_second = api_module._load_chunk_embeddings(arxiv_id, 1)
    finally:
        api_module._load_cited_chunks.cache_clear()
        api_module._load_chunk_embeddings.cache_clear()

    assert chunks_first is chunks_second
    assert embeddings_first is embeddings_second
    assert read_files == [(arxiv_id, "json"), (arxiv_id, "npy")]


def test_classify_endpoint_allows_missing_cited_paragraphs(client, monkeypatch) -> None:
    captured_request = {}

    def fake_predict(citing_sentence, **kwargs):
        captured_request["citing_sentence"] = citing_sentence
        captured_request.update(kwargs)
        return {"predicted_label": "Evidence"}

    monkeypatch.setattr(api_module, "predict", fake_predict)
    response = client.post(
        "/api/v1/classify-citation",
        json={
            "citing_sentence": "A citation [CITA].",
            "type_model": "scibert",
            "use_chunks": False,
        },
    )

    assert response.status_code == 200
    assert captured_request["cited_paragraphs"] is None
    assert captured_request["use_chunks"] is False


def test_bert_model_is_not_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_load_model_if_needed",
        lambda: pytest.fail("No debe cargarse un modelo para una selección inválida."),
    )

    with pytest.raises(ValueError, match="'scibert'"):
        predict(citing_sentence="A citation sentence.", type_model="bert")


def test_scibert_citation_marker_matches_training_format(monkeypatch) -> None:
    captured_context = []

    class FakeModel:
        def __call__(self, **_inputs):
            return type(
                "FakeOutputs",
                (),
                {"logits": torch.tensor([[1.0, 0.0]])},
            )()

    monkeypatch.setattr(api_module, "_load_model_if_needed", lambda: None)
    monkeypatch.setattr(api_module, "_tokenizer_scibert", object())
    monkeypatch.setattr(api_module, "_model_scibert", FakeModel())
    monkeypatch.setattr(
        api_module,
        "_id2label_scibert",
        {0: "Background", 1: "Evidence"},
    )
    monkeypatch.setattr(
        api_module,
        "_build_scibert_inputs",
        lambda context, *_args: (
            captured_context.append(context) or {}
        ),
    )

    predict(
        citing_sentence="The cited work supports the result [CITA].",
        type_model="scibert",
    )

    assert captured_context == [
        "The cited work supports the result #CITATION_TAG."
    ]


def test_scibert_can_classify_without_chunks(monkeypatch) -> None:
    captured_chunks = []

    class FakeModel:
        def __call__(self, **_inputs):
            return type(
                "FakeOutputs",
                (),
                {"logits": torch.tensor([[1.0, 0.0]])},
            )()

    monkeypatch.setattr(
        api_module,
        "_load_cited_chunks",
        lambda *_args: pytest.fail("No debe cargarse el caché si se desactivan los chunks."),
    )
    monkeypatch.setattr(api_module, "_load_model_if_needed", lambda: None)
    monkeypatch.setattr(api_module, "_tokenizer_scibert", object())
    monkeypatch.setattr(api_module, "_model_scibert", FakeModel())
    monkeypatch.setattr(
        api_module,
        "_id2label_scibert",
        {0: "Background", 1: "Evidence"},
    )
    monkeypatch.setattr(
        api_module,
        "_build_scibert_inputs",
        lambda _context, chunks, *_args: (captured_chunks.extend(chunks) or {}),
    )

    result = predict(
        citing_sentence="The cited work supports the result [CITA].",
        type_model="scibert",
        use_chunks=False,
    )

    assert captured_chunks == []
    assert result["chunks"] == []


@pytest.mark.parametrize(
    ("type_model", "classifier_name"),
    [
        ("gemma3", "_classify_with_gemma3"),
        ("nvidia_nemotron_3_ultra_550b_a55b", "_classify_with_nvidia"),
    ],
)
def test_external_models_can_classify_without_chunks(
    monkeypatch,
    type_model,
    classifier_name,
) -> None:
    received_chunks = []

    def fake_classifier(sentence, chunks):
        received_chunks.append(chunks)
        return {
            "predicted_label": "Evidence",
            "confidence": 0.8,
            "citing_sentence": sentence,
            "all_probabilities": {},
            "model_name": "Test model",
        }

    monkeypatch.setattr(
        api_module,
        "_load_cited_chunks",
        lambda *_args: pytest.fail("No debe cargarse el caché si se desactivan los chunks."),
    )
    monkeypatch.setattr(api_module, classifier_name, fake_classifier)

    result = predict(
        citing_sentence="The cited work supports the result [CITA].",
        type_model=type_model,
        use_chunks=False,
    )

    assert received_chunks == [[]]
    assert result["chunks"] == []


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


def test_rank_cited_chunks_uses_cached_specter2_embeddings(monkeypatch) -> None:
    query_embedding = torch.zeros((1, 768))
    query_embedding[0, 0] = 1.0
    encoded_texts = []

    def fake_embed_texts(texts, tokenizer, model, device):
        encoded_texts.extend(texts)
        return query_embedding

    monkeypatch.setattr(api_module, "_embed_specter2_texts", fake_embed_texts)

    chunks = [
        {"text": "Less similar", "embedding_text": "Less similar", "section": "Intro"},
        {"text": "Most similar", "embedding_text": "Most similar", "section": "Methods"},
        {"text": "Opposite", "embedding_text": "Opposite", "section": "Results"},
    ]
    cached_embeddings = np.zeros((3, 768), dtype=np.float32)
    cached_embeddings[0, 1] = 1.0
    cached_embeddings[1, 0] = 1.0
    cached_embeddings[2, 0] = -1.0

    ranked_chunks = api_module._rank_cited_chunks(
        "Citation sentence",
        chunks,
        cached_embeddings,
        tokenizer=object(),
        model=object(),
        device="cpu",
    )

    assert [chunk["text"] for chunk, _ in ranked_chunks] == [
        "Most similar",
        "Less similar",
        "Opposite",
    ]
    assert encoded_texts == ["Citation sentence"]


def test_rank_cited_chunks_encodes_missing_cache_with_specter2(monkeypatch) -> None:
    query_embedding = torch.zeros((1, 768))
    query_embedding[0, 0] = 1.0
    chunk_embedding = torch.zeros((1, 768))
    chunk_embedding[0, 0] = 1.0
    encoded_batches = []

    def fake_embed_texts(texts, tokenizer, model, device):
        encoded_batches.append(texts)
        return query_embedding if len(encoded_batches) == 1 else chunk_embedding

    monkeypatch.setattr(api_module, "_embed_specter2_texts", fake_embed_texts)
    chunks = [
        {
            "text": "Visible paragraph",
            "embedding_text": "Original paragraph",
            "section": "Introduction",
        }
    ]

    ranked_chunks = api_module._rank_cited_chunks(
        "Citation sentence",
        chunks,
        chunk_embeddings=None,
        tokenizer=object(),
        model=object(),
        device="cpu",
    )

    assert ranked_chunks[0][0]["text"] == "Visible paragraph"
    assert encoded_batches == [["Citation sentence"], ["Original paragraph"]]


def test_scibert_input_matches_finetuning_sequence(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "MAX_LENGTH", 12)

    class FakeTokenizer:
        cls_token_id = 101
        sep_token_id = 102

        def __init__(self):
            self.encoded_texts = []
            self.ids = {
                "The claim [CITA]": [1, 2],
                "Intro: First paragraph": [11, 12],
                "Method: Second paragraph": [13],
                "Results: Third paragraph": [14, 15, 16],
            }

        def encode(self, text, add_special_tokens):
            self.encoded_texts.append((text, add_special_tokens))
            return self.ids[text]

    tokenizer = FakeTokenizer()
    ranked_chunks = [
        ({"section": "Intro", "text": "First paragraph"}, 0.9),
        ({"section": "Method", "text": "Second paragraph"}, 0.8),
        ({"section": "Results", "text": "Third paragraph"}, 0.7),
    ]

    inputs = api_module._build_scibert_inputs(
        "The claim [CITA]",
        ranked_chunks,
        tokenizer,
        "cpu",
    )

    assert inputs["input_ids"].tolist() == [
        [101, 1, 2, 102, 11, 12, 102, 13, 102, 14, 15, 102]
    ]
    assert inputs["token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
    assert tokenizer.encoded_texts == [
        ("The claim [CITA]", False),
        ("Intro: First paragraph", False),
        ("Method: Second paragraph", False),
        ("Results: Third paragraph", False),
    ]


def test_gemma3_prompt_and_response_use_catalog_categories(monkeypatch) -> None:
    captured_request = {}
    category = "Evidence"
    response_body = json.dumps(
        {
            "response": json.dumps(
                {"function": category, "confidence": 0.82}
            )
        }
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return response_body

    def fake_urlopen(request, timeout):
        captured_request["url"] = request.full_url
        captured_request["body"] = json.loads(request.data)
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(api_module.urllib.request, "urlopen", fake_urlopen)

    result = api_module._classify_with_gemma3(
        "This supports our findings [CITA].",
        [
            (
                {
                    "section": "Results",
                    "text": "The method improved the measured outcome.",
                },
                0.91,
            ),
            (
                {
                    "section": "Discussion",
                    "text": "The findings support the hypothesis.",
                },
                0.82,
            ),
            (
                {
                    "section": "Introduction",
                    "text": "Prior work established the research context.",
                },
                0.73,
            ),
            (
                {"section": "Ignored", "text": "Only the top three are included."},
                0.1,
            ),
        ],
    )

    assert result["predicted_label"] == category
    assert result["confidence"] == 0.82
    assert result["all_probabilities"] == {}
    assert result["model_name"] == "Gemma 3"
    assert captured_request["body"]["model"] == "gemma3:12b"
    assert "Name: Evidence" in captured_request["body"]["prompt"]
    assert "This supports our findings [CITA]." in captured_request["body"]["prompt"]
    assert captured_request["body"]["format"] == "json"
    prompt = captured_request["body"]["prompt"]
    assert "Section: Results\nText: The method improved the measured outcome." in prompt
    assert "Section: Discussion\nText: The findings support the hypothesis." in prompt
    assert "Section: Introduction\nText: Prior work established the research context." in prompt
    assert "Only the top three are included." not in prompt


def test_gemma3_selection_does_not_load_scibert(monkeypatch) -> None:
    captured_sentence = []
    monkeypatch.setattr(
        api_module,
        "_load_model_if_needed",
        lambda: pytest.fail("Gemma 3 no debe cargar el modelo SciBERT."),
    )
    monkeypatch.setattr(
        api_module,
        "_classify_with_gemma3",
        lambda sentence, _ranked_chunks=None: (
            captured_sentence.append(sentence)
            or {
                "predicted_label": "Gap",
                "confidence": 0.7,
                "citing_sentence": sentence,
                "all_probabilities": {},
                "model_name": "Gemma 3",
            }
        ),
    )

    result = predict(
        citing_sentence="Prior work leaves an open question [CITA].",
        type_model="gemma3",
    )

    assert result["predicted_label"] == "Gap"
    assert result["model_name"] == "Gemma 3"
    assert captured_sentence == ["Prior work leaves an open question [CITA]."]


def test_nvidia_model_uses_gemma_prompt_and_configured_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module.settings,
        "api_key_nvidia",
        SecretStr("test-nvidia-api-key"),
    )
    captured_request = {}
    response_text = json.dumps({"function": "Evidence", "confidence": 0.91})
    content_chunks = [response_text[:16], response_text[16:]]
    response_lines = [
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": content}}]}
        )
        for content in content_chunks
    ]
    response_lines.append("data: [DONE]")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode):
            assert decode_unicode
            return iter(response_lines)

    def fake_post(url, headers, json, stream, timeout):
        captured_request["url"] = url
        captured_request["body"] = json
        captured_request["headers"] = headers
        captured_request["stream"] = stream
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(api_module.requests, "post", fake_post)
    ranked_chunks = [
        ({"section": "Results", "text": "The results support the claim."}, 0.9)
    ]

    result = api_module._classify_with_nvidia(
        "The results confirm the hypothesis [CITA].",
        ranked_chunks,
    )

    assert captured_request["url"] == api_module.NVIDIA_API_URL
    assert captured_request["headers"]["Authorization"] == "Bearer test-nvidia-api-key"
    assert captured_request["headers"]["Accept"] == "text/event-stream"
    assert captured_request["stream"] is True
    assert captured_request["timeout"] == (
        api_module.NVIDIA_CONNECT_TIMEOUT_SECONDS,
        api_module.NVIDIA_TIMEOUT_SECONDS,
    )
    assert captured_request["body"]["max_tokens"] == 16384
    assert captured_request["body"]["seed"] == 0
    assert captured_request["body"]["reasoning_effort"] == "low"
    assert captured_request["body"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert captured_request["body"]["messages"][0]["content"] == (
        api_module._build_gemma3_prompt(
            "The results confirm the hypothesis [CITA].",
            ranked_chunks,
        )
    )
    assert result["predicted_label"] == "Evidence"
    assert result["confidence"] == 0.91
    assert result["model_name"] == "Nemotron 3 Ultra 550B (NVIDIA)"
    assert "test-nvidia-api-key" not in json.dumps(result)


def test_nvidia_model_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "api_key_nvidia", None)
    monkeypatch.setattr(
        api_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("No debe solicitarse sin API key."),
    )

    with pytest.raises(api_module.ModelServiceError, match="API_KEY_NVIDIA"):
        api_module._classify_with_nvidia("A claim [CITA].")


def test_nvidia_connection_errors_are_reported(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module.settings,
        "api_key_nvidia",
        SecretStr("test-nvidia-api-key"),
    )
    def raise_connection_error(*_args, **_kwargs):
        raise api_module.requests.ConnectionError("connection refused")

    monkeypatch.setattr(api_module.requests, "post", raise_connection_error)

    with pytest.raises(api_module.ModelServiceError, match="conexión con NVIDIA API") as error:
        api_module._classify_with_nvidia("A claim [CITA].")

    assert error.value.status_code == 503


def test_nvidia_read_timeout_returns_gateway_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module.settings,
        "api_key_nvidia",
        SecretStr("test-nvidia-api-key"),
    )

    def raise_read_timeout(*_args, **_kwargs):
        raise api_module.requests.ReadTimeout("read timed out")

    monkeypatch.setattr(api_module.requests, "post", raise_read_timeout)

    with pytest.raises(api_module.ModelServiceError, match="completar la respuesta") as error:
        api_module._classify_with_nvidia("A claim [CITA].")

    assert error.value.status_code == 504


def test_nvidia_selection_does_not_load_scibert(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_load_model_if_needed",
        lambda: pytest.fail("NVIDIA no debe cargar el modelo SciBERT."),
    )
    monkeypatch.setattr(
        api_module,
        "_classify_with_nvidia",
        lambda sentence, _ranked_chunks=None: {
            "predicted_label": "Comparison",
            "confidence": 0.75,
            "citing_sentence": sentence,
            "all_probabilities": {},
            "model_name": "Nemotron 3 Ultra 550B (NVIDIA)",
        },
    )

    result = predict(
        citing_sentence="We compare these results with [CITA].",
        type_model="nvidia_nemotron_3_ultra_550b_a55b",
    )

    assert result["predicted_label"] == "Comparison"
    assert result["model_name"] == "Nemotron 3 Ultra 550B (NVIDIA)"