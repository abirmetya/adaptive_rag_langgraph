from typing import Any, Dict, Literal
from graph.state import AdaptiveRAGState
from graph.chains.retrieval_grader import retrieval_grader_chain

def retrieval_grader(state: AdaptiveRAGState) -> Literal["content_generation", "rewrite_query"]:
    "Check if the retrieved documents are relevant to the user's question"
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    response = retrieval_grader_chain.invoke({"query": query, "retrieved_docs": retrieved_docs})
    if response.is_relevant == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        return "content_generation"
    if response.is_relevant == "no":
        print("---GRADE: DOCUMENT NOT RELEVANT-- REWRITE THE QUERY---")
        return "rewrite_query"