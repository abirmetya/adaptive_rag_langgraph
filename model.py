from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
load_dotenv()

## Initialize LLM model
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.0, max_tokens = 100)

## Initialize Embedding model
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")