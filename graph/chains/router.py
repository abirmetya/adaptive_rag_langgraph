from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Literal
from pydantic import BaseModel, Field
from model import llm


# Create uery Router Chain. Query Router will provide either of the three outputs ['vectorstore', 'websearch', 'simple']. 

## Create a pydantic model for the query router structure output
class QueryRouterOutput(BaseModel):
    "Route a user query to most relevant datasource"
    datasource: Literal["vectorstore","websearch"] = Field(
        ..., description="Given a user query, route to the most relevant datasource. The options are 'vectorstore', 'websearch'"
    )

structured_llm_router = llm.with_structured_output(QueryRouterOutput)

system_prompt = """You are a query router to route user query to Vectorstore or websearch or simple. Vectorstore contains about agents, prompt engineering, adversial attacks. Use the vectorstore if the query is related to these topics. Use websearch if the query is not related to these topics."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "query: {query}"),
    ]
)

query_router_chain = router_prompt | structured_llm_router

if __name__ == "__main__":
    # Example usage of the query router chain
    query_router_chain.invoke({"query": 'Hey, how are you?'})  # Should return 'vectorstore'