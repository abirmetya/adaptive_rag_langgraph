from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import Document
from graph.state import AdaptiveRAGState


def websearch_route(state: AdaptiveRAGState):
    "Perform web search to retrieve documents"
    # Placeholder for web search logic
    # For now, we will return an empty list
    query = state["query"]
    print(query)
    search_tool = DuckDuckGoSearchRun()
    print(search_tool)
    results = search_tool.invoke(str(query))  # Perform web search
    print(results)
    web_results = Document(page_content=results, metadata={"source": "websearch"})
    return {"retrieved_docs": web_results}