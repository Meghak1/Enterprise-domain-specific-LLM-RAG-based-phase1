from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "C://Enterprise-domain-specific-LLM-RAG-based-phase1//data//semiconductor//Intel_architecture253665-089-sdm-vol-1.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} text chunks")

# Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector database
vectorstore.save_local("vector_store")

print("Vector store created and saved")
