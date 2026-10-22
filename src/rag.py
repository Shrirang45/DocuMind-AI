
from typing import List, Dict, Any

from langchain_ollama import OllamaLLM
from src.data_ingestion import EmbeddingsManager, VectorStore


# ============================================================
# LOCAL LLM CONFIGURATION
# ============================================================

llm = OllamaLLM(
    model="mistral",
    temperature=0.2,
    num_predict=768,  # Allows longer answers than 256 tokens
    num_ctx=4096,     # More room for retrieved context and answers
)


# ============================================================
# RETRIEVAL
# ============================================================

class RAGRetriever:
    """Retrieve relevant document chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingsManager,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically relevant chunks."""

        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_docs = []

            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]

            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            ):
                if not document:
                    continue

                # Assumes the Chroma collection uses cosine distance.
                similarity_score = 1.0 - float(distance)

                if similarity_score < score_threshold:
                    continue

                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata or {},
                        "similarity_score": similarity_score,
                        "distance": float(distance),
                        "rank": rank,
                    }
                )

            return retrieved_docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []


# ============================================================
# CONTEXT RELEVANCE CHECK
# ============================================================

def is_context_relevant(query: str, context: str, model) -> bool:
    """
    Check whether the retrieved text supports the question.
    Fail closed if the relevance check fails.
    """

    prompt = f"""
You are a strict document relevance checker.

Decide whether the DOCUMENT CONTEXT contains information that
directly answers, explains, or meaningfully supports the QUESTION.

Rules:
- Judge the meaning, not just exact word matches.
- A shared topic alone is not enough.
- Do not use outside knowledge.
- If the context contains useful information for the question,
  reply YES.
- If the context is unrelated or insufficient, reply NO.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

Reply with exactly one word: YES or NO.
"""

    try:
        verdict = model.invoke(prompt).strip().upper()
        verdict = verdict.strip(" .!\"'\n")

        return verdict == "YES"

    except Exception as e:
        print(f"Context relevance check failed: {e}")
        return False


# ============================================================
# GENERAL KNOWLEDGE FALLBACK
# ============================================================

def generate_general_answer(query: str, model) -> str:
    """Generate a clearly labelled answer outside document grounding."""

    prompt = f"""
Answer the question using your general knowledge.

Rules:
- Be accurate, clear, and helpful.
- Explain important steps and give an example when requested.
- Do not claim that the answer comes from uploaded documents.
- If you are unsure, say so.

QUESTION:
{query}

ANSWER:
"""

    try:
        answer = model.invoke(prompt).strip()

        return (
            "I couldn't find relevant information about this question "
            "in your uploaded documents.\n\n"
            "**General answer (not based on your documents):**\n\n"
            + answer
        )

    except Exception as e:
        print(f"General answer generation error: {e}")
        return (
            "I couldn't find relevant information in your uploaded "
            "documents, and I couldn't generate a general answer."
        )


# ============================================================
# PAGE NUMBER HELPERS
# ============================================================

def get_display_page(metadata: Dict[str, Any]) -> Any:
    """
    PyPDFLoader normally stores page indexes starting at zero.
    Convert them to human-readable page numbers starting at one.
    """

    page = metadata.get("page")

    if page is None:
        return "unknown"

    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return page


# ============================================================
# ANSWER GENERATION
# ============================================================

def rag(
    query: str,
    retriever: RAGRetriever,
    model=llm,
    top_k: int = 5,
    min_score: float = 0.35,
    return_context: bool = False,
) -> Dict[str, Any]:
    """
    Answer using retrieved document context when relevant.
    Otherwise, provide a clearly labelled general answer.
    """

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        score_threshold=min_score,
    )

    # --------------------------------------------------------
    # No chunks passed the retrieval threshold
    # --------------------------------------------------------

    if not results:
        output = {
            "answer": generate_general_answer(query, model),
            "sources": [],
            "confidence": 0.0,
        }

        if return_context:
            output["context"] = ""

        return output

    # --------------------------------------------------------
    # Build context from retrieved chunks
    # --------------------------------------------------------

    context_parts = []

    for doc in results:
        metadata = doc.get("metadata") or {}
        source = metadata.get(
            "source_file",
            metadata.get("source", "unknown"),
        )
        page = get_display_page(metadata)

        context_parts.append(
            f"[Document: {source} | Page: {page}]\n"
            f"{doc['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # --------------------------------------------------------
    # Verify that context is relevant
    # --------------------------------------------------------

    if not is_context_relevant(query, context, model):
        output = {
            "answer": generate_general_answer(query, model),
            "sources": [],
            "confidence": 0.0,
        }

        if return_context:
            output["context"] = ""

        return output

    # --------------------------------------------------------
    # Generate a complete, document-grounded answer
    # --------------------------------------------------------

    prompt = f"""
You are DocuMind AI, an assistant that answers questions using
uploaded PDF documents.

Answer the user's question using ONLY the DOCUMENT CONTEXT below.

IMPORTANT RULES:
1. Answer the actual question directly.
2. If the user asks to explain a concept, provide a clear explanation.
3. If the user asks for an example, finish the example completely.
4. If equations or calculations are present, show the relevant steps.
5. Use all relevant context provided, not just the first passage.
6. Do not stop halfway through a sentence, equation, list, or example.
7. Do not invent facts that are not supported by the context.
8. If the context supports only part of the question, explain that
   limitation instead of inventing the missing information.
9. Do not state which page contains the answer in the answer text.
   Page numbers will be displayed separately in the source list.
10. Do not claim to have used information that is absent from context.
11. Be detailed enough to answer the question, but avoid repetition.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Write the complete answer below:
"""

    try:
        response = model.invoke(prompt).strip()

        if not response:
            response = (
                "The model did not generate an answer. "
                "Please try asking the question again."
            )

    except Exception as e:
        print(f"Answer generation error: {e}")
        response = (
            "Sorry, I couldn't generate an answer from the documents."
        )

    # --------------------------------------------------------
    # Prepare source references
    # --------------------------------------------------------

    sources = []
    seen_sources = set()

    for doc in results:
        metadata = doc.get("metadata") or {}

        source_name = metadata.get(
            "source_file",
            metadata.get("source", "unknown"),
        )
        page = get_display_page(metadata)

        source_key = (source_name, page)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "source": source_name,
                "page": page,
                "score": doc["similarity_score"],
                "preview": doc["content"][:300]
                + ("..." if len(doc["content"]) > 300 else ""),
            }
        )

    output = {
        "answer": response,
        "sources": sources,
        "confidence": max(
            doc["similarity_score"] for doc in results
        ),
    }

    if return_context:
        output["context"] = context

    return output
