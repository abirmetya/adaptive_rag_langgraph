from model import llm
from langchain import hub

# llm_generation = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.0)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm