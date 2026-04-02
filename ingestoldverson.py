from langchain_community.document_loaders import PyPDFLoader #used to read and extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter #used to break long documents into smaller chunks
from langchain_openai import OpenAIEmbeddings #converts text into numerical vectors
from langchain_community.vectorstores import FAISS #used to store and search embeddings efficiently

#intel architecture sdm volume 1 pdf from 13 volumes, has basic understanding
PDF_PATH = "C://Enterprise-domain-specific-LLM-RAG-based-phase1//data//semiconductor//Intel_architecture253665-089-sdm-vol-1.pdf"

#load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load() # Read all pages from the PDF into Document objects

print(f"Loaded {len(documents)} pages")

#split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, #breaking documents as per 1000 chrs
    chunk_overlap=200 # to capture contextual info
)
#split full documents into smaller chunks
docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} text chunks")

#create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings) # Convert chunks into vectors and store in FAISS index

#save vector database
vectorstore.save_local("vector_store")

print("Vector store created and saved")
