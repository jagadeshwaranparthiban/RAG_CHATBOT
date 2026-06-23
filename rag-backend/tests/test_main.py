"""
Tests for the RAG backend.

All tests mock out Qdrant and Groq so the suite runs without any live
services — important for CI, where neither is available.

Run locally:  pytest -v
"""

import io
from unittest.mock import MagicMock, patch

import docx
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

import main
from main import app, chunk_text, read_file_text

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — build real in-memory file bytes so parser tests are genuine
# ---------------------------------------------------------------------------

def make_txt(content: str = "Hello from a text file.") -> bytes:
    return content.encode("utf-8")


def make_pdf(content: str = "Hello from a PDF.") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, content)
    c.save()
    return buf.getvalue()


def make_docx(content: str = "Hello from a DOCX.") -> bytes:
    buf = io.BytesIO()
    doc = docx.Document()
    doc.add_paragraph(content)
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_ok(self):
        res = client.get("/health")
        assert res.status_code == 200

    def test_response_shape(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "groq_model" in data
        assert "qdrant_collection" in data
        assert "qdrant_url" in data

    def test_status_value(self):
        assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# read_file_text — parser
# ---------------------------------------------------------------------------

class TestReadFileText:
    def test_txt_extraction(self):
        text = read_file_text("sample.txt", make_txt("Parsing plain text."))
        assert "Parsing plain text." in text

    def test_pdf_extraction(self):
        text = read_file_text("sample.pdf", make_pdf("Parsing a PDF file."))
        assert "Parsing a PDF file." in text

    def test_docx_extraction(self):
        text = read_file_text("sample.docx", make_docx("Parsing a DOCX file."))
        assert "Parsing a DOCX file." in text

    def test_unsupported_extension_raises_http_400(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            read_file_text("file.csv", b"col1,col2\n1,2")
        assert exc_info.value.status_code == 400

    def test_txt_utf8_decoding(self):
        # Non-ASCII characters should be preserved, not lost
        text = read_file_text("unicode.txt", "Tamil: தமிழ்".encode("utf-8"))
        assert "தமிழ்" in text


# ---------------------------------------------------------------------------
# chunk_text — chunker
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n  ") == []

    def test_short_text_returns_single_chunk(self):
        result = chunk_text("short text", chunk_size=800)
        assert result == ["short text"]

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 400          # ~2000 characters
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_overlap(self):
        # Every word in the original text should appear in at least one chunk
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 30
        chunks = chunk_text(text, chunk_size=100, overlap=30)
        combined = " ".join(chunks)
        for word in ["alpha", "gamma", "kappa"]:
            assert word in combined

    def test_no_empty_chunks(self):
        text = "word " * 300
        assert all(len(c.strip()) > 0 for c in chunk_text(text))

    def test_chunk_size_respected(self):
        text = "x" * 2000
        for chunk in chunk_text(text, chunk_size=200, overlap=20):
            assert len(chunk) <= 200


# ---------------------------------------------------------------------------
# /ingest endpoint
# ---------------------------------------------------------------------------

def _mock_qdrant_for_ingest():
    """Returns a context manager that patches qdrant to accept ingest calls."""
    mock = MagicMock()
    mock.collection_exists.return_value = False
    mock.get_embedding_size.return_value = 384
    return patch.object(main, "qdrant", mock)


class TestIngestEndpoint:
    def test_rejects_empty_file_list(self):
        # FastAPI returns 422 (Unprocessable Entity) when a required field
        # is missing from the form — not 400, which would be our own error.
        res = client.post("/ingest")
        assert res.status_code == 422

    def test_single_txt_file(self):
        with _mock_qdrant_for_ingest():
            res = client.post(
                "/ingest",
                files=[("files", ("note.txt", make_txt("FastAPI is great."), "text/plain"))],
            )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["total_chunks"] >= 1

    def test_single_pdf_file(self):
        with _mock_qdrant_for_ingest():
            res = client.post(
                "/ingest",
                files=[("files", ("report.pdf", make_pdf("PDF content here."), "application/pdf"))],
            )
        assert res.status_code == 200
        assert res.json()["total_chunks"] >= 1

    def test_single_docx_file(self):
        with _mock_qdrant_for_ingest():
            res = client.post(
                "/ingest",
                files=[("files", ("doc.docx", make_docx("DOCX content here."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))],
            )
        assert res.status_code == 200
        assert res.json()["total_chunks"] >= 1

    def test_multiple_files_in_one_request(self):
        with _mock_qdrant_for_ingest():
            res = client.post(
                "/ingest",
                files=[
                    ("files", ("a.txt", make_txt("File A content."), "text/plain")),
                    ("files", ("b.txt", make_txt("File B content."), "text/plain")),
                ],
            )
        data = res.json()
        assert res.status_code == 200
        assert len(data["files"]) == 2

    def test_response_contains_filename_and_chunk_count(self):
        with _mock_qdrant_for_ingest():
            res = client.post(
                "/ingest",
                files=[("files", ("myfile.txt", make_txt("some content"), "text/plain"))],
            )
        file_result = res.json()["files"][0]
        assert "filename" in file_result
        assert "chunks_indexed" in file_result


# ---------------------------------------------------------------------------
# /query endpoint
# ---------------------------------------------------------------------------

def _fake_search_result(text="Qdrant stores vectors.", source="doc.txt", score=0.9):
    point = MagicMock()
    point.payload = {"text": text, "source": source, "chunk_index": 0}
    point.score = score
    result = MagicMock()
    result.points = [point]
    return result


def _fake_groq_response(answer="Here is the answer."):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=answer))]
    return response


class TestQueryEndpoint:
    def test_missing_groq_key_returns_500(self):
        with patch.object(main.settings, "groq_api_key", ""), \
             patch.object(main.qdrant, "collection_exists", return_value=True):
            res = client.post("/query", json={"question": "What is RAG?"})
        assert res.status_code == 500
        assert "GROQ_API_KEY" in res.json()["detail"]

    def test_no_collection_returns_400(self):
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=False):
            res = client.post("/query", json={"question": "What is RAG?"})
        assert res.status_code == 400
        assert "ingested" in res.json()["detail"]

    def test_successful_query_returns_answer_and_sources(self):
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=True), \
             patch.object(main.qdrant, "query_points", return_value=_fake_search_result()), \
             patch.object(main.groq_client.chat.completions, "create", return_value=_fake_groq_response("The answer is 42.")):
            res = client.post("/query", json={"question": "What is the answer?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "The answer is 42."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "doc.txt"

    def test_no_matching_chunks_returns_friendly_message(self):
        empty = MagicMock()
        empty.points = []
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=True), \
             patch.object(main.qdrant, "query_points", return_value=empty):
            res = client.post("/query", json={"question": "totally unrelated"})
        assert res.status_code == 200
        assert res.json()["sources"] == []

    def test_top_k_parameter_is_forwarded(self):
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=True), \
             patch.object(main.qdrant, "query_points", return_value=_fake_search_result()) as mock_qp, \
             patch.object(main.groq_client.chat.completions, "create", return_value=_fake_groq_response()):
            client.post("/query", json={"question": "test", "top_k": 3})
        # The limit passed to each Prefetch should be top_k * 4 = 12
        call_kwargs = mock_qp.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_sources_include_score_rounded_to_4dp(self):
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=True), \
             patch.object(main.qdrant, "query_points", return_value=_fake_search_result(score=0.876543)), \
             patch.object(main.groq_client.chat.completions, "create", return_value=_fake_groq_response()):
            res = client.post("/query", json={"question": "test"})
        assert res.json()["sources"][0]["score"] == 0.8765

    def test_groq_error_returns_502(self):
        from groq import GroqError
        with patch.object(main.settings, "groq_api_key", "fake-key"), \
             patch.object(main.qdrant, "collection_exists", return_value=True), \
             patch.object(main.qdrant, "query_points", return_value=_fake_search_result()), \
             patch.object(main.groq_client.chat.completions, "create", side_effect=GroqError("rate limited")):
            res = client.post("/query", json={"question": "test"})
        assert res.status_code == 502