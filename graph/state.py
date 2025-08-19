from typing import List,TypedDict

class AdaptiveRAGState(TypedDict):
    "State for the Adaptive RAG graph"
    query: str
    datasource: str
    retrieved_docs: List[str]
    is_relevant: str
    quality_checker: str
    llm_response: str