import streamlit as st #for UI
from langchain.chains import RetrievalQA #this chain handles retrieval and prompt building and LLM answering
from langchain_community.vectorstores import FAISS #used to load the saved vector database
from langchain_community.embeddings import HuggingFaceEmbeddings #converts text queries into numerical vectors
from langchain_community.llms import Ollama #used to run local LLM models.........like llama3

#config
VECTOR_DB_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"


st.set_page_config(page_title="Semiconductor Assistent", layout="wide")
st.title("Enterprise domain specific LLM RAG based")

st.write(
    "Ask questions about Intel processors, architecture, caches, memory, "
    "instruction sets, and more. Answers are grounded only in uploaded documents."
)

#load embeddings (MUST match ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

#load vector store
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

#load local LLM
llm = Ollama(model=LLM_MODEL)

#create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

#UI input
query = st.text_input("Ask a question about Intel architecture")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query) # to run the qa chain with user query

        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("Sources used"): # to display source document pages
            for doc in result["source_documents"]:
                st.write(f"- Page {doc.metadata.get('page', 'N/A')}")
