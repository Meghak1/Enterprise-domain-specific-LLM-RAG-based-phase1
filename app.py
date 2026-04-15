import os
import fitz
import logging
import subprocess
import time
import gradio as gr
import pyttsx3
import matplotlib.pyplot as plt

from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Configuration class to store file paths
@dataclass
class Config:
    PDF_PATH: str = "C://Enterprise-domain-specific-LLM-RAG-based-phase1//data//semiconductor//Intel_architecture253665-089-sdm-vol-1.pdf"
    VECTOR_DB_PATH: str = "vector_store"

# Initialize configuration
config = Config()

# Set logging level
logging.basicConfig(level=logging.INFO)

# Load PDF document
PDF_DOC = fitz.open(config.PDF_PATH)

# Cache dictionary to store rendered pages for faster reuse
PAGE_CACHE = {}

# Initialize embedding model for converting text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database from local storage
vectorstore = FAISS.load_local(
    config.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever with top-k results set to 6
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Initialize LLM using Ollama with phi3 model
llm = Ollama(model="phi3")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def speak_text(text):
    engine.say(text)  # Queue the text to be spoken
    engine.runAndWait()  # Execute speech

# Function to render a PDF page as an image
def render_page(page_num):
    # Check if page already exists in cache
    if page_num in PAGE_CACHE:
        return PAGE_CACHE[page_num]

    try:
        # Get page from PDF
        page = PDF_DOC[page_num]

        # Convert page to image with scaling for better clarity
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))

        # Save image locally
        img_path = f"page_{page_num}.png"
        pix.save(img_path)

        # Store in cache
        PAGE_CACHE[page_num] = img_path

        return img_path
    except:
        # Return None if rendering fails
        return None

# Function to open a specific page in external PDF viewer
def open_pdf_page(page_num):
    try:
        # Increment page number since viewer starts from 1
        page_num = int(page_num) + 1

        # Open PDF in Adobe Acrobat at specific page
        subprocess.Popen([
            "C:\\Program Files\\Adobe\\Acrobat DC\\Acrobat\\Acrobat.exe",
            "/A", f"page={page_num}",
            config.PDF_PATH
        ])
    except:
        # Fallback to default PDF opener
        os.startfile(config.PDF_PATH)

# Function to compute evaluation metrics for RAG output
def compute_metrics(query, answer, docs):
    try:
        # Convert query and answer into embeddings
        query_emb = embeddings.embed_query(query)
        answer_emb = embeddings.embed_query(answer)

        # Convert each document chunk into embeddings
        doc_embs = [embeddings.embed_query(d.page_content) for d in docs]

        # Combine all document text into a single context
        context_text = " ".join([d.page_content for d in docs])

        # Generate embedding for combined context
        context_emb = embeddings.embed_query(context_text)

        # Compute similarity between answer and context
        faithfulness = cosine_similarity([answer_emb], [context_emb])[0][0]

        # Compute similarity between query and each document chunk
        relevances = [
            cosine_similarity([query_emb], [doc_emb])[0][0]
            for doc_emb in doc_embs
        ]

        # Average relevance score
        avg_relevance = sum(relevances) / len(relevances)

        # Compute similarity between query and answer
        answer_relevance = cosine_similarity([query_emb], [answer_emb])[0][0]

        # Return rounded metric values
        return {
            "faithfulness": round(float(faithfulness), 3),
            "context_relevance": round(float(avg_relevance), 3),
            "answer_relevance": round(float(answer_relevance), 3)
        }

    except:
        # Return default values if any error occurs
        return {
            "faithfulness": 0,
            "context_relevance": 0,
            "answer_relevance": 0
        }

# Function to plot metric values as a bar chart
def plot_metrics(metrics):
    # Extract metric names and values
    names = list(metrics.keys())
    values = list(metrics.values())

    # Create bar chart
    plt.figure()
    plt.bar(names, values)
    plt.ylim(0, 1)  # Set y-axis range

    # Save chart as image
    path = "metrics.png"
    plt.savefig(path)
    plt.close()

    return path

# Main function to handle query processing using RAG
def ask(query):
    # Start timer
    start_time = time.time()

    # Retrieve relevant document chunks
    docs = retriever.invoke(query)

    # Filter out very small or weak chunks
    filtered_docs = []
    for d in docs:
        if len(d.page_content.strip()) > 100:
            filtered_docs.append(d)

    # Keep top 3 chunks after filtering
    docs = filtered_docs[:3]

    # Record retrieval time
    retrieval_time = time.time()

    # Combine retrieved chunks into context
    context = "\n\n".join([d.page_content for d in docs])

    # Create prompt for LLM
    prompt = f"""
Answer ONLY from the context below.
If answer not found, say "Not found in document".
Mention exact page numbers.

Context:
{context}

Question:
{query}
"""

    # Generate answer using LLM
    answer = llm.invoke(prompt)

    # Record end time
    end_time = time.time()

    # Calculate base timing metrics
    base_metrics = {
        "response_time": round(end_time - start_time, 2),
        "retrieval_time": round(retrieval_time - start_time, 2),
        "generation_time": round(end_time - retrieval_time, 2),
        "chunks": len(docs)
    }

    # Compute RAG-specific metrics
    rag_metrics = compute_metrics(query, answer, docs)

    # Combine all metrics
    all_metrics = {**base_metrics, **rag_metrics}

    # Generate metrics graph
    graph_path = plot_metrics(rag_metrics)

    # Prepare image and info outputs
    images = []
    info = []

    # Loop through retrieved documents
    for d in docs:
        # Get page number from metadata
        page = d.metadata.get("page", 0)

        # Render page as image
        img = render_page(page)

        if img:
            # Create short snippet for display
            snippet = d.page_content[:150].replace("\n", " ")
            images.append((img, f"Page {page} | {snippet}..."))

        # Store detailed info
        info.append({
            "page": page,
            "content": d.page_content[:500]
        })

    return answer, images, info, all_metrics, graph_path

# Build Gradio UI
with gr.Blocks(css="""
body {background: black; color: white;}
button {background: #0ea5e9 !important; color: white;}
""") as demo:

    # Title
    gr.Markdown("## Grounded RAG based LLM Assistant for Semiconductor Industry")

    # Input textbox for user query
    query = gr.Textbox(label="Ask Question")

    # Output textbox for answer
    answer_box = gr.Textbox(label="Answer")

    # Button to trigger text-to-speech
    speak_btn = gr.Button("Speak")

    # Gallery to display source pages
    gallery = gr.Gallery(label="Source Pages")

    # JSON display for page details
    page_info = gr.JSON(label="Source Details")

    # JSON display for metrics
    metrics_box = gr.JSON(label="Metrics")

    # Image display for graph
    graph_output = gr.Image(label="Metrics Graph")

    # Button to process query
    btn = gr.Button("Ask")

    # Function to process query
    def process(q):
        return ask(q)

    # Function to speak answer
    def speak(ans):
        speak_text(ans)

    # Function to open selected page from gallery
    def open_from_gallery(evt: gr.SelectData):
        caption = evt.value[1]
        page = int(caption.split()[2])  # extract page number
        open_pdf_page(page)

    # Bind button click to process function
    btn.click(
        process,
        inputs=query,
        outputs=[answer_box, gallery, page_info, metrics_box, graph_output]
    )

    # Bind speak button
    speak_btn.click(speak, inputs=answer_box, outputs=[])

    # Enable clicking on gallery images
    gallery.select(open_from_gallery)

# Launch the application
demo.launch()