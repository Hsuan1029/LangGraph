#Test embeddings
#prints a length and five numbers, it worked.
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2:1b")
embeddings = OllamaEmbeddings(model="all-minilm")

vector = embeddings.embed_query("What skills does Winston Taylor look for?")
print(len(vector), vector[:5])