# Create document ingetion pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from model import embed_model

## We will provide a list of URLs to the WebBaseLoader to load the documents from the web.
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/AI_agent",
    "https://en.wikipedia.org/wiki/Prompt_engineering",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist] ## Flatten the list of lists

## We are using tiktoken encoding to ensure accurate 250 token chunks for the LLM.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=10)

doc_split = text_splitter.split_documents(docs_list)  # Split the documents into chunks

## Create vectore store with documents
vector_store = Chroma.from_documents(
    documents = doc_split, 
    embedding = embed_model, 
    collection_name="adaptive_rag", 
    persist_directory="./.chroma"
    )

# Create retriever from the vector store
retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # Retrieve top 3 most similar documents
)

if __name__ == "__main__":
    ## Test the retriever
    retriever.get_relevant_documents("What is AI agents?")  # Should return a list of documents related to the query