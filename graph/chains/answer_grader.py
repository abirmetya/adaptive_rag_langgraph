from model import llm
from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Let's create a Answer quality checker chain. It will check if the LLM response is of good quality or not. The output will be either 'yes' or 'no'.

class AnswerQualityCheckerOutput(BaseModel):
    "Check if the LLM response is of good quality or not"
    is_good_quality: Literal["yes", "no"] = Field(
        ..., description="Check if the LLM response is of good quality or not. The options are 'yes' or 'no'."
    )

answer_quality_checker = llm.with_structured_output(AnswerQualityCheckerOutput)

system_prompt = """ You are an answer quality checker wheather the LLM response correctly answers the user's question. If the LLM response is of good quality, return 'yes'. If the LLM response is not of good quality, return 'no'."""

answer_quality_checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "query: {query}\nllm_response: {llm_response}"),
    ])

answer_quality_checker_chain = answer_quality_checker_prompt | answer_quality_checker


if __name__ == "__main__":
    # Example usage of the answer quality checker chain
    answer_quality_checker_chain.invoke({"query": "What is the capital of France?",
                                     "llm_response": "The capital of France is Paris."})  # Should return