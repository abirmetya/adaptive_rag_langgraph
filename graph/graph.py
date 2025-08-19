from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

from graph.state import AdaptiveRAGState

from graph.chains.answer_grader import answer_quality_checker_chain
from graph.chains.hallucination_grader import hallucination_checker_chain
from graph.chains.router import query_router_chain
from graph.chains.rewrite_query import rewrite_query_chain

from graph.nodes.generate import content_generation
from graph.nodes.grade_documents import retrieval_grader
from graph.nodes.retrieve import retrieval_route
from graph.nodes.web_search import websearch_route

from graph.state import AdaptiveRAGState


def query_router(state: AdaptiveRAGState):
    "Route the user query to the most relevant datasource"
    query = state["query"]
    response = query_router_chain.invoke({"query": query})
    # print(response.datasource)
    return {"datasource": response.datasource}

def query_router_decision(state: AdaptiveRAGState) -> Literal["retrieval_route", "websearch_route"]:
    "Route the user query to the most relevant datasource"
    datasource = state["datasource"]
    print("---ROUTE QUESTION---")
    if datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieval_route"
    if datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch_route"

def rewrite_query(state: AdaptiveRAGState):
    "Rewrite the user query to be more specific and clear"
    query = state["query"]
    response = rewrite_query_chain.invoke({"query": query})
    return {"query": response}  # Update the query in the state with the rewritten query

def hallucination_checker(state: AdaptiveRAGState):
    "Check if the LLM response is hallucinated or not"
    retrieved_docs = state["retrieved_docs"]
    llm_response = state["llm_response"]
    response = hallucination_checker_chain.invoke({"retrieved_docs": retrieved_docs, "llm_response": llm_response})
    if response.is_hallucinated == "yes":
        return "not_approved"
    if response.is_hallucinated == "no":
        return "approved"
    
def answer_quality_checker(state: AdaptiveRAGState):
    "Check if the LLM response is of good quality or not"
    query = state["query"]
    llm_response = state["llm_response"]
    response = answer_quality_checker_chain.invoke({"query": query, "llm_response": llm_response})
    return {"quality_checker": response}

def endnote(state: AdaptiveRAGState):
    quality_checker = state["quality_checker"]
    if quality_checker == "yes":
        return "END"
    if quality_checker == "no":
        return "rewritequery"
    
# Create Graph
graph = StateGraph(AdaptiveRAGState)

# Create node
graph.add_node("query_router",query_router)
graph.add_node("retrieval_route", retrieval_route)
graph.add_node("websearch_route", websearch_route)
graph.add_node("content_generation", content_generation)
graph.add_node("rewrite_query", rewrite_query)
# graph.add_node("hallucination_checker", hallucination_checker)
# graph.add_node("answer_quality_checker", answer_quality_checker)
# graph.add_node("retrieval_grader", retrieval_grader)


# Create edge
graph.add_edge(START, "query_router")
graph.add_conditional_edges("query_router", query_router_decision)
# graph.add_conditional_edges(START, query_router)
# graph.add_edge(query_router, "retrieval_route")
graph.add_conditional_edges("retrieval_route", retrieval_grader)
graph.add_conditional_edges("content_generation", hallucination_checker, {"not_approved": "content_generation", "approved": END})

# graph.add_conditional_edges("content_generation", hallucination_checker, {"not_approved": "content_generation", "approved": "answer_quality_checker"})
# graph.add_conditional_edges("answer_quality_checker",endnote,{"END": END, "rewritequery": "rewrite_query"})

# graph.add_edge(content_generation,"answer_quality_checker")

graph.add_edge("rewrite_query","retrieval_route")


# graph.add_edge(query_router, "websearch_route")
graph.add_edge("websearch_route", "content_generation")
graph.add_edge("content_generation", END)

# Complie
workflow = graph.compile()


if __name__ == "__main__":    
    # Run
    print(workflow.invoke({"query": "What is AI agents?"}))