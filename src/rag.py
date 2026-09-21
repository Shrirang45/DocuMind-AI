
from typing import List, Dict, Any

from langchain_ollama import OllamaLLM

from src.data_ingestion import EmbeddingsManager, VectorStore


class RAGRetriever:
    """Retrieve relevant document chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingsManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks above the cosine similarity threshold."""

        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]

            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1
            ):
                # ChromaDB cosine distance = 1 - cosine similarity
                similarity_score = 1.0 - float(distance)

                if similarity_score < score_threshold:
                    continue

                retrieved_docs.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata or {},
                    "similarity_score": similarity_score,
                    "distance": float(distance),
                    "rank": rank
                })

            return retrieved_docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []


def is_context_relevant(query: str, context: str, llm) -> bool:
    """
    Ask the local LLM whether the context contains information
    that can actually answer the user's question.
    Fail closed if the relevance check fails.
    """

    prompt = f"""
You are a strict document relevance checker.

Determine whether the DOCUMENT CONTEXT contains information
that directly answers, explains, or meaningfully supports
the QUESTION.

A shared topic alone is NOT enough.
Do not use outside knowledge.
If the context is unrelated or insufficient, answer NO.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

Reply with exactly one word:
YES or NO
"""

    try:
        verdict = llm.invoke(prompt).strip().upper()
        verdict = verdict.strip(" .!\"'\n")

        return verdict == "YES"

    except Exception as e:
        print(f"Context relevance check failed: {e}")
        return False


def generate_general_answer(query: str, llm) -> str:
    """Answer without using or citing uploaded documents."""

    prompt = f"""
Answer the following question using your general knowledge.

Do not claim that the answer comes from uploaded documents.
If you are unsure, clearly say so.
Be accurate, clear, and concise.

Question: {query}

Answer:
"""

    try:
        answer = llm.invoke(prompt).strip()

        return (
            "I couldn't find relevant information about this question "
            "in your uploaded documents.\n\n"
            "**General answer (not based on your documents):**\n\n"
            + answer
        )

    except Exception:
        return (
            "I couldn't find relevant information about this question "
            "in your uploaded documents. I couldn't generate a general "
            "answer either."
        )


def rag(
    query,
    retriever,
    llm,
    top_k=3,
    min_score=0.35,
    return_context=False
):
    """Answer from relevant documents, otherwise use general knowledge."""

    results = retriever.retrieve(
        query,
        top_k=top_k,
        score_threshold=min_score
    )

    # No chunks passed the retrieval threshold
    if not results:
        output = {
            "answer": generate_general_answer(query, llm),
            "sources": [],
            "confidence": 0.0
        }

        if return_context:
            output["context"] = ""

        return output

    context = "\n\n".join(
        doc["content"] for doc in results
    )

    # Retrieval similarity alone is not enough.
    # Verify that the chunks actually relate to the question.
    if not is_context_relevant(query, context, llm):
        output = {
            "answer": generate_general_answer(query, llm),
            "sources": [],
            "confidence": 0.0
        }

        if return_context:
            output["context"] = ""

        return output

    prompt = f"""
Answer the question using ONLY the document context below.

Rules:
- Do not introduce unsupported facts.
- If the context does not contain the answer, say so.
- Be clear and concise.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = llm.invoke(prompt).strip()
    except Exception as e:
        print(f"Answer generation error: {e}")
        response = "Sorry, I couldn't generate an answer from the documents."

    sources = []

    # Sources are returned only after the context passes
    # the relevance check.
    seen_sources = set()

    for doc in results:
        metadata = doc.get("metadata") or {}

        source_name = metadata.get(
            "source_file",
            metadata.get("source", "unknown")
        )

        page = metadata.get("page", "unknown")

        # Avoid duplicate filename/page citations
        source_key = (source_name, page)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append({
            "source": source_name,
            "page": page,
            "score": doc["similarity_score"],
            "preview": doc["content"][:300] + "..."
        })

    output = {
        "answer": response,
        "sources": sources,
        "confidence": max(
            doc["similarity_score"] for doc in results
        )
    }

    if return_context:
        output["context"] = context

    return output


llm = OllamaLLM(
    model="mistral",
    temperature=0.2,
    num_predict=256,
    num_ctx=2048
)