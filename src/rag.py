from typing import List, Dict, Any

from langchain_ollama import OllamaLLM

from src.data_ingestion import EmbeddingsManager, VectorStore

class RAGRetriever:
    "Handles query based retrieval form vector store"

    def __init__(self,vector_store:VectorStore,embedding_manager:EmbeddingsManager):
        """
        ARGS:
        vector_store : vector store contains document embeddings
        embedding manager : MAnager for generating query embeddings"""

        self.vector_store=vector_store
        self.embedding_manager=embedding_manager

    def retrieve(self, query : str, top_k:int=5,score_threshold:float=0.0)-> List[Dict[str,Any]]:
        """retrive relevant document for a query
        query: the search query
        top_k: Number of top result to return
        score threshold: minimum similarity score threshold
        
        returns :
            list of dictionaries containing retrieved document and metadata"""

        print(f'Retrieving document for query: {query}')
        print(f'top K : {top_k}, score threshold: {score_threshold}')

        #generating query embedding
        query_embedding=self.embedding_manager.generate_embeddings([query])[0]

        #search in vector store
        try:
            results=self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            #process results
            retrieved_docs=[]

            if results['documents'] and results['documents'][0]:
                documents=results['documents'][0]
                metadatas=results['metadatas'][0]
                distances=results['distances'][0]
                ids=results['ids'][0]

                for i, (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    #convert distance to similarity score(chromadb uses cosine distance)
                    similarity_score=1-distance
                    if similarity_score>=score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i+1
                        })
                print(f'Retrieved {len(retrieved_docs)} documents (after filtering)')

            else:
                print("No document found")
            return retrieved_docs
        except Exception as e:
            print(f"error during retrieval: {e}")
            return []



def rag(query,retriever,llm,top_k=3,min_score=0.2,return_context=False):
    ##retrieve the context
    results=retriever.retrieve(query,top_k=top_k,score_threshold=min_score)
    if not results:
        return {'answer':"No relevant context found to answer question.",'sources':[],'confidence':0.0,'context':''}
    
    context="\n\n".join([doc['content'] for doc in results])
    sources= [{
        'source':doc['metadata'].get('source_file',doc['metadata'].get('source','unknown')),
        'page': doc['metadata'].get('page','unknown'),
        'score':doc['similarity_score'],
        'preview':doc['content'][:300] + '...'
    } for doc in results]

    confidence=max([doc['similarity_score']for doc in results])
    #generate answer using llm
    prompt=f"""Use the following context to answer the question precisely.
     Context:
        {context} 
     Question: {query}
     Answer: """
    

    response=llm.invoke(prompt)
    output={
        'answer':response,
        'sources':sources,
        'confidence':confidence
    }

    if return_context:
        output['context']=context
    return output

llm=OllamaLLM(model='mistral')
