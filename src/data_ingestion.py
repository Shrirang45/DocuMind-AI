"""Reliable PDF ingestion for DocuMind AI.

Features:
- Paths are resolved relative to the project root, not the current directory.
- Embedding model downloads on first use and is saved locally for offline use.
- PDF pages are stored with 1-based page numbers.
- SHA-256 hashes distinguish unchanged and modified files.
- Existing indexed chunks are preserved until a replacement has been parsed,
  chunked, embedded, and written successfully.
- Only files in the configured upload directory are synchronized.

Compatibility: exposes EmbeddingsManager, VectorStore, DocumentChunker,
process_all_pdf(), and ingest_documents().
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
_BATCH_SIZE = 64
_INGEST_LOCK = threading.RLock()


def get_file_hash(file_path: os.PathLike | str) -> str:
    """Return the SHA-256 hash of a file, reading it in bounded memory."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be simple scalar values."""
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


# ---------------------------------------------------------------------------
# PDF loading and chunking
# ---------------------------------------------------------------------------

def load_pdf(pdf_file: os.PathLike | str) -> List[Document]:
    """Load a PDF; attach stable filename/hash metadata and 1-based pages."""
    path = Path(pdf_file).resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    file_hash = get_file_hash(path)
    pages = PyPDFLoader(str(path)).load()
    for doc in pages:
        # PyPDFLoader's page metadata is normally zero-based.
        raw_page = doc.metadata.get("page")
        try:
            page_number = int(raw_page) + 1 if raw_page is not None else None
        except (TypeError, ValueError):
            page_number = None
        doc.metadata.update({
            "source_file": path.name,
            "source": str(path),
            "file_hash": file_hash,
            "file_type": "pdf",
        })
        if page_number is not None:
            doc.metadata["page"] = page_number
            doc.metadata["page_number"] = page_number
    return pages


def process_all_pdf(pdf_directory: os.PathLike | str) -> List[Document]:
    """Load PDFs from a directory recursively; report individual failures."""
    folder = Path(pdf_directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    output: List[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        try:
            docs = load_pdf(path)
            if not docs:
                logger.warning("No readable pages in %s", path.name)
            output.extend(docs)
        except Exception:
            logger.exception("Failed to read PDF %s", path)
    return output


class DocumentChunker:
    """Split pages into overlapping chunks while retaining page metadata."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []
        return self.text_splitter.split_documents(list(documents))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingsManager:
    """Load a local embedding model, or download it once if missing."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_path = Path(model_name).expanduser() if model_name else DEFAULT_MODEL_DIR
        if not self.model_path.is_absolute():
            self.model_path = (PROJECT_ROOT / self.model_path).resolve()
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            # A saved SentenceTransformer model contains modules.json.
            if self.model_path.is_dir() and (self.model_path / "modules.json").is_file():
                logger.info("Loading local embedding model from %s", self.model_path)
                self.model = SentenceTransformer(str(self.model_path), local_files_only=True)
            else:
                logger.info("Downloading embedding model %s (first run requires internet)", MODEL_ID)
                model = SentenceTransformer(MODEL_ID)
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.model_path.with_name(self.model_path.name + ".tmp")
                if temp_path.exists():
                    import shutil
                    shutil.rmtree(temp_path, ignore_errors=True)
                model.save(str(temp_path))
                # Replace the destination only after a complete save.
                if self.model_path.exists():
                    import shutil
                    shutil.rmtree(self.model_path)
                temp_path.rename(self.model_path)
                self.model = SentenceTransformer(str(self.model_path), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model. First run needs internet, or provide a valid local model at {self.model_path}."
            ) from exc

    def generate_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model has not loaded.")
        if not texts:
            return np.empty((0, self.model.get_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=_BATCH_SIZE,
        )
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Chroma vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Persistent ChromaDB wrapper used by the RAG retriever."""

    def __init__(self, collection_name: str = COLLECTION_NAME, persist_directory: Optional[os.PathLike | str] = None):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory or DEFAULT_VECTOR_DIR).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        # Note: Chroma cannot change the distance metric of an existing collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "PDF chunks for DocuMind AI"},
        )

    def get_stored_source_files(self) -> set[str]:
        result = self.collection.get(include=["metadatas"])
        names: set[str] = set()
        for meta in result.get("metadatas") or []:
            if meta and meta.get("source_file"):
                names.add(str(meta["source_file"]))
        return names

    def get_file_hashes(self) -> Dict[str, set[str]]:
        result = self.collection.get(include=["metadatas"])
        mapping: Dict[str, set[str]] = {}
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            name = meta.get("source_file")
            file_hash = meta.get("file_hash")
            if name and file_hash:
                mapping.setdefault(str(name).casefold(), set()).add(str(file_hash))
        return mapping

    def is_document_indexed(self, source_file: str, file_hash: str) -> bool:
        # Use case-insensitive filename matching in Python, then exact hash.
        result = self.collection.get(include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            if str(meta.get("source_file", "")).casefold() == source_file.casefold() and meta.get("file_hash") == file_hash:
                return True
        return False

    def delete_document(self, source_file: str) -> None:
        """Delete all chunks for one exact stored filename."""
        self.collection.delete(where={"source_file": source_file})

    def delete_file_version(self, source_file: str, file_hash: str) -> None:
        """Delete only a specific indexed version of a PDF."""
        self.collection.delete(
            where={
                "$and": [
                    {"source_file": source_file},
                    {"file_hash": file_hash},
                ]
            }
        )

    def add_documents(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        """Upsert chunks. Caller should remove old chunks only after this succeeds."""
        if len(documents) != len(embeddings):
            raise ValueError("Document count and embedding count differ.")
        if not documents:
            return

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        vectors: List[List[float]] = []
        for doc, vector in zip(documents, embeddings):
            meta = dict(doc.metadata)
            source_name = str(meta.get("source_file", "unknown"))
            file_hash = str(meta.get("file_hash", "unknown"))
            page = str(meta.get("page", "unknown"))
            start = str(meta.get("start_index", ""))
            content = doc.page_content
            # Include hash, page and chunk content in the ID; avoids collisions
            # when separate PDFs contain the same chunk text.
            identity = f"{source_name.casefold()}|{file_hash}|{page}|{start}|{content}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            meta["chunk_id"] = chunk_id
            ids.append(chunk_id)
            texts.append(content)
            metadatas.append(_safe_metadata(meta))
            vectors.append(np.asarray(vector, dtype=np.float32).tolist())

        # Batch writes to avoid oversized requests.
        for start in range(0, len(ids), 256):
            end = start + 256
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )


# ---------------------------------------------------------------------------
# Failure-safe ingestion
# ---------------------------------------------------------------------------

def ingest_documents(pdf_directory: Optional[os.PathLike | str] = None) -> Tuple[VectorStore, EmbeddingsManager]:
    """Index new/changed PDFs without deleting the old version prematurely.

    Returns (vector_store, embedding_manager) for compatibility with the app.
    Files are keyed case-insensitively. A failed parse/embed/write leaves the
    prior indexed version in place. The vector store is scoped to the supplied
    upload directory; do not point it at a mixed-purpose folder.
    """
    upload_dir = Path(pdf_directory or DEFAULT_UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    with _INGEST_LOCK:
        store = VectorStore()
        embedder = EmbeddingsManager()

        # Top-level PDFs only, matching Streamlit upload-folder expectations.
        files = sorted(
            p for p in upload_dir.iterdir()
            if p.is_file() and p.suffix.casefold() == ".pdf"
        )
        current = {p.name.casefold(): p for p in files}
        stored_names = store.get_stored_source_files()
        stored_by_key = {name.casefold(): name for name in stored_names}
        stored_hashes = store.get_file_hashes()

        # Parse/index each new or modified file. Old chunks remain available
        # until the new chunks have been written successfully.
        for key, pdf_path in current.items():
            try:
                file_hash = get_file_hash(pdf_path)
                old_name = stored_by_key.get(key)
                if old_name and old_name == pdf_path.name and store.is_document_indexed(pdf_path.name, file_hash):
                    logger.info("Skipping unchanged PDF: %s", pdf_path.name)
                    continue

                logger.info("Indexing new/modified PDF: %s", pdf_path.name)
                pages = load_pdf(pdf_path)
                if not pages:
                    raise ValueError("PDF produced no readable pages; previous index preserved.")

                chunks = DocumentChunker().split_documents(pages)
                if not chunks:
                    raise ValueError("PDF produced no text chunks; previous index preserved.")

                texts = [chunk.page_content for chunk in chunks]
                embeddings = embedder.generate_embeddings(texts)
                # Write new chunks first. IDs include the new hash, so these
                # are distinct from the old version until cleanup succeeds.
                store.add_documents(chunks, embeddings)

                # Only after successful upsert remove the previous version.
                # Delete by old hash, not filename alone, so newly written
                # chunks for the replacement cannot be deleted accidentally.
                if old_name:
                    for old_hash in stored_hashes.get(key, set()):
                        if old_hash != file_hash:
                            store.delete_file_version(old_name, old_hash)
                    if old_name != pdf_path.name:
                        # Filename casing changed; remove any remaining chunks
                        # under the old spelling after the new version is saved.
                        store.delete_document(old_name)

                logger.info("Successfully indexed %s (%d chunks)", pdf_path.name, len(chunks))

            except Exception:
                logger.exception("Failed to ingest %s; preserving previous index when possible", pdf_path.name)

        # Clean up indexed PDFs that are no longer in the upload directory.
        # This only targets source filenames that were previously indexed here.
        for key, stored_name in stored_by_key.items():
            if key not in current:
                try:
                    store.delete_document(stored_name)
                    logger.info("Removed deleted upload from index: %s", stored_name)
                except Exception:
                    logger.exception("Failed to remove stale index for %s", stored_name)

        return store, embedder
