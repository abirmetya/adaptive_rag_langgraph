from ingestion import retriever
from graph.state import AdaptiveRAGState

def retrieval_route(state: AdaptiveRAGState):
    "Retrieve documents from the vector store"
    query = state["query"]
    print("---RETRIEVE---")
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_docs_list = [doc.page_content for doc in retrieved_docs]
    return {"retrieved_docs": retrieved_docs_list}