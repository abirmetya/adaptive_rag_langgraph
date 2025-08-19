from langchain.prompts import ChatPromptTemplate
from model import llm
from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Now, let's create a retrieval grader chain. It will check for whether the retrieved documents are actually relevant to the user's question. The output will be either 'yes' or 'no'.

class RetrievalGraderOutput(BaseModel):
    "Check if the retrieved documents are relevant to the user's question"
    is_relevant: Literal["yes", "no"] = Field(
        ..., description="Check if the retrieved documents are relevant to the user's question. The options are 'yes' or 'no'."
    )

retrieval_grader = llm.with_structured_output(RetrievalGraderOutput)

system_prompt = """You are a retrieval grader to check if the retrieved documents are relevant to the user's question. If the retrieved documents are relevant, return 'yes'. If the retrieved documents are not relevant, return 'no'."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "query: {query}\nretrieved_docs: {retrieved_docs}"),
    ]
)

retrieval_grader_chain = grader_prompt | retrieval_grader

if __name__ == "__main__":
        
    # Example usage of the retrieval grader chain
    retrieval_grader_chain.invoke({
        "query": "What is the capital of France?",
        "retrieved_docs": "AI agents are autonomous programs that can perform tasks without human intervention. They can be used for a variety of purposes, such as data analysis, web scraping, and automation."})  # Should return 'no'
