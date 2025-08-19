from langchain.prompts import ChatPromptTemplate
from model import llm



system_prompt = """You are a query rewriter to rewrite the user's query to be more specific and clear. The rewritten query should be more focused and should help in retrieving more relevant documents. Do not write any additional introductory text or greetings."""
rewrite_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "query: {query}")
    ]
)
rewrite_query_chain = rewrite_query_prompt | llm