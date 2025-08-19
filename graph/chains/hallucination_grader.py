from model import llm
from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Now let's create a Hallucination Checker Chain. It will check if the LLM response is hallucinated or not. The output will be either 'yes' or 'no'.

class HallucinationCheckerOutput(BaseModel):
    "Check if the LLM response is hallucinated or not"
    is_hallucinated: Literal["yes", "no"] = Field(
        ..., description="Check if the LLM response is hallucinated or not. The options are 'yes' or 'no'."
    )

hallucination_checker = llm.with_structured_output(HallucinationCheckerOutput)

system_prompt = """You are a hallucination checker to check if the LLM response is hallucinated or not. If the LLM response is hallucinated, return 'yes'. If the LLM response is not hallucinated, and supported by a set of reteieved facts, return 'no'."""

hallucination_checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "retrieved_docs: {retrieved_docs}\nllm_response: {llm_response}"),
    ]
)

hallucination_checker_chain = hallucination_checker_prompt | hallucination_checker


if __name__ == "__main__":    
    # Example usage of the hallucination checker chain
    hallucination_checker_chain.invoke({
        "retrieved_docs": "AI agents are autonomous programs that can perform tasks without human intervention. They can be used for a variety of purposes, such as data analysis, web scraping, and automation.",
        "llm_response": "The capital of France is Paris."})  # Should return 'yes'