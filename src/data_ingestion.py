
import hashlib
from pathlib import Path
from typing import Any, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

COLLECTION_NAME = "pdf_documents"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# FILE HASHING
# ============================================================

def get_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf(pdf_file: Path) -> List[Document]:
    """Load a PDF and attach its source, hash, and 1-based page."""

    file_hash = get_file_hash(pdf_file)
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    for document in documents:
        document.metadata["source_file"] = pdf_file.name
        document.metadata["file_type"] = "pdf"
        document.metadata["file_hash"] = file_hash

        # PyPDFLoader uses 0-based indexes.
        # Store human-readable PDF page numbers instead.
        if "page" in document.metadata:
            page_number = int(document.metadata["page"]) + 1
            document.metadata["page"] = page_number
            document.metadata["page_number"] = page_number

    return documents


def process_all_pdf(pdf_directory) -> List[Document]:
    """Load all PDFs recursively from a directory."""

    pdf_dir = Path(pdf_directory)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        path for path in pdf_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )

    print(f"Found {len(pdf_files)} PDF files.")

    all_documents = []

    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            documents = load_pdf(pdf_file)
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages.")

        except Exception as error:
            print(f"Error processing {pdf_file.name}: {error}")

    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

class DocumentChunker:
    """Split PDF pages into smaller, overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller "
                "than chunk_size."
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents while preserving their metadata."""

        if not documents:
            print("No documents to split.")
            return []

        print(f"Original pages: {len(documents)}")

        chunks = self.text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks.")
        return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

class EmbeddingsManager:
    """Load the local embedding model or download it on first use."""

    def __init__(self, model_name: Optional[str] = None):
        self.model = None

        if model_name is None:
            self.model_path = DEFAULT_MODEL_DIR
        else:
            supplied_path = Path(model_name)
            if supplied_path.is_absolute():
                self.model_path = supplied_path
            else:
                self.model_path = PROJECT_ROOT / supplied_path

        self._load_model()

    def _load_model(self):
        """Load an existing model or download and save it."""

        try:
            if (
                self.model_path.is_dir()
                and any(self.model_path.iterdir())
            ):
                print(f"Loading local model: {self.model_path}")

                self.model = SentenceTransformer(
                    str(self.model_path),
                    local_files_only=True,
                )

            else:
                print(f"Downloading embedding model: {MODEL_ID}")

                self.model = SentenceTransformer(MODEL_ID)

                self.model_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                self.model.save(str(self.model_path))

                print(f"Model saved to: {self.model_path}")

            print(
                "Embedding model loaded. Dimension:",
                self.model.get_embedding_dimension(),
            )

        except Exception as error:
            raise RuntimeError(
                "Could not load the embedding model. On the first run, "
                "check your internet connection. For offline use, make "
                f"sure the model exists at: {self.model_path}"
            ) from error

    def generate_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Convert a list of texts into numerical vectors."""

        if self.model is None:
            raise ValueError("Embedding model is not loaded.")

        if not texts:
            return np.empty(
                (0, self.model.get_embedding_dimension()),
                dtype=np.float32,
            )

        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# ============================================================
# CHROMADB VECTOR STORE
# ============================================================

class VectorStore:
    """Store and manage document chunks and their embeddings."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory=None,
    ):
        self.collection_name = collection_name

        self.persist_directory = Path(
            persist_directory or DEFAULT_VECTOR_DIR
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document embeddings for RAG",
                "hnsw:space": "cosine",
            },
        )

        print(f"Vector collection: {self.collection_name}")
        print(f"Stored chunks: {self.collection.count()}")

    def is_document_indexed(
        self,
        source_file: str,
        file_hash: str,
    ) -> bool:
        """Return True if this exact file version is indexed."""

        results = self.collection.get(
            where={
                "$and": [
                    {"source_file": source_file},
                    {"file_hash": file_hash},
                ]
            },
            limit=1,
        )

        return bool(results.get("ids"))

    def delete_document(self, source_file: str):
        """Remove every stored chunk for a source PDF."""

        self.collection.delete(
            where={"source_file": source_file}
        )

        print(f"Removed indexed chunks for: {source_file}")

    def get_stored_source_files(self) -> set:
        """Return the distinct PDF filenames stored in ChromaDB."""

        results = self.collection.get(
            include=["metadatas"]
        )

        source_files = set()

        for metadata in results.get("metadatas") or []:
            if metadata:
                source_file = metadata.get("source_file")

                if source_file:
                    source_files.add(source_file)

        return source_files

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ):
        """Insert or update document chunks and their embeddings."""

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        if not documents:
            print("No chunks to add.")
            return

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for index, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            source = document.metadata.get("source_file", "unknown")
            page = document.metadata.get("page", 1)
            content = document.page_content

            # Stable ID for the same source, page, chunk position,
            # and content. Different chunks get different IDs.
            id_input = f"{source}|{page}|{index}|{content}"
            doc_id = hashlib.sha256(
                id_input.encode("utf-8")
            ).hexdigest()

            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["content_length"] = len(content)

            # ChromaDB metadata values must be scalar values.
            for key, value in list(metadata.items()):
                if value is None:
                    metadata[key] = ""
                elif isinstance(value, (list, dict, tuple)):
                    metadata[key] = str(value)

            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(content)
            embeddings_list.append(
                np.asarray(embedding).tolist()
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

        print(f"Indexed {len(documents)} chunks.")
        print(f"Total stored chunks: {self.collection.count()}")


# ============================================================
# INGESTION PIPELINE
# ============================================================

def ingest_documents(pdf_directory):
    """
    Synchronize uploaded PDFs with ChromaDB.

    - New PDFs are indexed.
    - Unchanged PDFs are skipped.
    - Modified PDFs are re-indexed.
    - Removed PDFs are removed from the vector store.
    - Filename comparisons ignore case.
    """

    pdf_dir = Path(pdf_directory).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = VectorStore()

    # The upload directory contains the PDFs managed by this app.
    pdf_files = sorted(
        path for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )

    print(f"\nFound {len(pdf_files)} uploaded PDFs.")

    current_by_key = {
        pdf_file.name.casefold(): pdf_file
        for pdf_file in pdf_files
    }

    stored_files = vectorstore.get_stored_source_files()

    stored_by_key = {
        name.casefold(): name
        for name in stored_files
    }

    # Remove indexed files that no longer exist in the upload folder.
    for key, stored_name in list(stored_by_key.items()):
        if key not in current_by_key:
            print(f"PDF no longer exists: {stored_name}")
            vectorstore.delete_document(stored_name)

    # Refresh stored filenames after cleanup.
    stored_files = vectorstore.get_stored_source_files()

    stored_by_key = {
        name.casefold(): name
        for name in stored_files
    }

    new_pdf_files = []

    for pdf_file in pdf_files:
        file_hash = get_file_hash(pdf_file)
        key = pdf_file.name.casefold()
        stored_name = stored_by_key.get(key)

        # If the file's spelling/case changed, refresh its metadata
        # so citations use the current filename.
        if (
            stored_name is not None
            and stored_name != pdf_file.name
        ):
            print(f"Filename changed: {stored_name} -> {pdf_file.name}")
            vectorstore.delete_document(stored_name)
            new_pdf_files.append(pdf_file)
            continue

        if vectorstore.is_document_indexed(
            pdf_file.name,
            file_hash,
        ):
            print(f"Skipping unchanged PDF: {pdf_file.name}")
            continue

        # A stored file with this name but a different hash is modified.
        if stored_name is not None:
            print(f"Modified PDF detected: {pdf_file.name}")
            vectorstore.delete_document(stored_name)
        else:
            print(f"New PDF detected: {pdf_file.name}")

        new_pdf_files.append(pdf_file)

    # Initialize the embedding manager because the caller may need it
    # for answering questions, even when all PDFs are unchanged.
    embedding_manager = EmbeddingsManager()

    if not new_pdf_files:
        print("No new or modified PDFs to ingest.")
        return vectorstore, embedding_manager

    documents = []

    for pdf_file in new_pdf_files:
        try:
            print(f"Loading: {pdf_file.name}")
            pdf_documents = load_pdf(pdf_file)

            if not pdf_documents:
                print(f"No readable pages found in {pdf_file.name}.")
                continue

            documents.extend(pdf_documents)
            print(f"Loaded {len(pdf_documents)} pages.")

        except Exception as error:
            print(f"Error loading {pdf_file.name}: {error}")

    if not documents:
        print("No readable PDF pages were found.")
        return vectorstore, embedding_manager

    # Split loaded pages into smaller chunks.
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.split_documents(documents)

    if not chunks:
        print("No text chunks were generated.")
        return vectorstore, embedding_manager

    # Generate vector embeddings for each chunk.
    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    # Save chunks, metadata, and embeddings to ChromaDB.
    vectorstore.add_documents(chunks, embeddings)

    print("PDF ingestion completed successfully.")

    return vectorstore, embedding_manager
