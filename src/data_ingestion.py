import os
import hashlib
from pathlib import Path
from typing import List, Any

import numpy as np
import chromadb

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

def get_file_hash(file_path):
    """Generate a SHA-256 hash for a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

## Read all the pdf inside the directory
def process_all_pdf(pdf_directory):
    "Process all PDF files in directory"
    all_documents=[]
    pdf_dir=Path(pdf_directory)

    #find all pdf files recursively
    pdf_files=list(pdf_dir.glob("**/*.pdf"))
    print(f'found {len(pdf_files)} PDF files to process')

    for pdf_file in pdf_files:
        print(f'\nProcessing: {pdf_file.name}')
        try:
            # Calculate fingerprint of the PDF
            file_hash = get_file_hash(pdf_file)

            loader=PyPDFLoader(str(pdf_file))
            documents=loader.load()

            #add source info to metadata
            for doc in documents:
                doc.metadata['source_file']=pdf_file.name
                doc.metadata['file_type']='pdf'
                doc.metadata['file_hash'] = file_hash
                # Convert PyPDFLoader's 0-based page number
                # to a normal 1-based PDF page number
                if "page" in doc.metadata:
                    doc.metadata["page"] = int(doc.metadata["page"]) + 1
            all_documents.extend(documents)
            print(f' Loaded {len(documents)} pages')
        except Exception as e:
            print(f' Error : {e}')

    print(f'\nTotal documents loaded: {len(all_documents)}')
    return all_documents


class DocumentChunker:
    """Splits loaded documents into smaller chunks."""

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List):
        """
        Split documents into smaller chunks.
        """

        print(f"Original documents: {len(documents)}")

        chunks = self.text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks")

        return chunks


class EmbeddingsManager:
    "Handles document Embeddings generation using sentence transformers"
    def __init__(
        self,
        model_name: str = "./models/all-MiniLM-L6-v2"): 

        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        "Load sentence transformer model"
        try:
            print(f'Loading embedding model {self.model_name}')
            self.model=SentenceTransformer(self.model_name,local_files_only=True)
            print(f'Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}')

        except Exception as e:
            print(f'error loading model {self.model_name}: {e }')
            raise   

    def generate_embeddings( self , texts : List[str]) -> np.ndarray:
        '''generates embeddings for a list of texts
        args:
            text: list of texts strings to embedd
        returns:
            numpy array of embeddings with shape (lenn(texts) , embedding_dim)
            '''

        if not self.model:
            raise ValueError("Model not loaded")

        print(f'generating embeddings for {len(texts)} texts')
        embeddings=self.model.encode(texts, show_progress_bar=True)
        print(f'generated embeddings with shape: {embeddings.shape}')
        return embeddings

class VectorStore:
    "Manages the document embeddings in a chromadb vector store"

    def __init__(self , collection_name: str='pdf_documents',persist_directory: str= "./data/vector_store"):
        """ 
        Initialize the vector store
        Args: 
            collection_name= name of the chromadb collection
            persist directory: Directory to persist the vector store
            """
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialize_store()

    def _initialize_store(self):
        "initialize chromadb client and collection"
        try: #create persistant chroma client
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)

            #get or create collection
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description" : "PDF document embeddings for RAG",
                          "hnsw:space": "cosine"}
            )
            print(f"vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f'Error initializing vector store: {e}')
            raise

    def is_document_indexed(self, source_file: str,file_hash:str) -> bool:
    
        """Check whether a document has already been indexed.
    Args:
        source_file: Name of the PDF file.
    Returns:
        True if the document exists in ChromaDB, otherwise False. """
    

        results = self.collection.get(
            where={
                "$and":[
                    {"source_file": source_file},
                    {"file_hash":file_hash}
                    ]
            },
            limit=1
        )

        return len(results["ids"]) > 0

    def delete_document(self, source_file: str):
        """
        Delete all chunks belonging to a specific PDF
        from ChromaDB.
        """

        print(f"Deleting document from vector store: {source_file}")

        self.collection.delete(
            where={"source_file": source_file}
        )

        print(
            f"Deleted chunks for: {source_file}"
        )

        print(
            f"Total documents remaining: {self.collection.count()}"
        )

    def get_stored_source_files(self):
        """
        Get all PDF filenames currently stored in ChromaDB.
        """

        results = self.collection.get(
            include=["metadatas"]
        )

        source_files = set()

        for metadata in results["metadatas"]:
            source_file = metadata.get("source_file")

            if source_file:
                source_files.add(source_file)

        return source_files

    def add_documents(self, documents: List[Any] , embeddings = np.ndarray):
        """
        Add documents and their embeddings to vector store
        Args: 
            documents =  List of langchain documents
            embeddings= corresponding embeddings of documents"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f'adding {len(documents)} documents to vector store ...')

        #prepare data for Chromadb
        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", 0)
            content = doc.page_content

            doc_id = hashlib.md5(
                                f"{source}_{page}_{content}".encode("utf-8")
                                ).hexdigest()
            ids.append(doc_id)

            #prepare metadeta
            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            #document content
            documents_text.append(doc.page_content)
            #embeddings
            embeddings_list.append(embedding.tolist())

        try: 
            self.collection.upsert (
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
            )
            print(f'Successfully added {len(documents)} documents to vector store')
            print(f'total documents in collection: {self.collection.count()}')
        except Exception as e:
            print(f'error adding documents in vector store : {e}')
            raise
def ingest_documents(pdf_directory):
    vectorstore = VectorStore()
    pdf_dir = Path(pdf_directory)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Only process PDFs directly inside the user uploads folder
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    current_files = {pdf_file.name for pdf_file in pdf_files}
    stored_files = vectorstore.get_stored_source_files()

    # Remove ChromaDB chunks for documents deleted from uploads
    deleted_files = stored_files - current_files
    for source_file in deleted_files:
        print(f"Document deleted by user: {source_file}")
        vectorstore.delete_document(source_file)

    new_pdf_files = []

    # Detect new and modified documents using SHA-256
    for pdf_file in pdf_files:
        file_hash = get_file_hash(pdf_file)

        if vectorstore.is_document_indexed(pdf_file.name, file_hash):
            print(f"Skipping unchanged document: {pdf_file.name}")
        else:
            print(f"New or modified document: {pdf_file.name}")

            if pdf_file.name in stored_files:
                vectorstore.delete_document(pdf_file.name)

            new_pdf_files.append(pdf_file)

    # Nothing new to process
    if not new_pdf_files:
        print("No new or modified documents to ingest.")
        embedding_manager = EmbeddingsManager()
        return vectorstore, embedding_manager

    documents = []

    # Load new or modified PDFs
    for pdf_file in new_pdf_files:
        print(f"Processing: {pdf_file.name}")

        try:
            file_hash = get_file_hash(pdf_file)
            loader = PyPDFLoader(str(pdf_file))
            pdf_documents = loader.load()

            for doc in pdf_documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
                doc.metadata["file_hash"] = file_hash

            documents.extend(pdf_documents)
            print(f"Loaded {len(pdf_documents)} pages")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    # Chunk documents
    document_chunking = DocumentChunker()
    chunks = document_chunking.split_documents(documents)

    # Generate embeddings
    embedding_manager = EmbeddingsManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store chunks and embeddings in ChromaDB
    vectorstore.add_documents(chunks, embeddings)

    return vectorstore, embedding_manager