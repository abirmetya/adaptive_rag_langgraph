from graph.state import AdaptiveRAGState
from graph.chains.generation import generation_chain

def content_generation(state: AdaptiveRAGState):
    "Generate content using the LLM"
    retrieved_docs = state["retrieved_docs"]
    query = state["query"]
    try:
        context = "\n".join(retrieved_docs)
    except TypeError:
        context = str(retrieved_docs)
    llm_response = generation_chain.invoke({"context": context, "question": query})
    return {"llm_response": llm_response}