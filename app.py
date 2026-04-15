import os
import fitz
import time
import gradio as gr
import pyttsx3
import speech_recognition as sr
import matplotlib.pyplot as plt
import threading

from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


# ================= CONFIG =================
# Store file paths
@dataclass
class Config:
    PDF_PATH: str = "C://Enterprise-domain-specific-LLM-RAG-based-phase1//data//semiconductor//Intel_architecture253665-089-sdm-vol-1.pdf"
    VECTOR_DB_PATH: str = "vector_store"

config = Config()

# Load PDF
PDF_DOC = fitz.open(config.PDF_PATH)

# Cache for rendered pages
PAGE_CACHE = {}

# Store chat history
chat_history = []


# ================= EMBEDDINGS =================
# Convert text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vectorstore = FAISS.load_local(
    config.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever for top-k documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ================= LLM =================
# Load local LLM model
llm = Ollama(model="phi3")


# ================= TEXT TO SPEECH =================
# Initialize engine
engine = pyttsx3.init()

# Set voice properties
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

# Internal function for speaking (runs in thread)
def _speak(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()

# Wrapper function to avoid blocking UI
def speak_text(text):
    print("Speaking:", text)
    thread = threading.Thread(target=_speak, args=(text,))
    thread.start()

# Stop speaking
def stop_speaking():
    engine.stop()


# ================= SPEECH TO TEXT =================
recognizer = sr.Recognizer()

def voice_input():
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
    except:
        return "Voice input failed"


# ================= PDF RENDER =================
# Convert PDF page to image and optionally highlight text
def render_page(page_num, highlight_text=None):
    page = PDF_DOC[page_num]

    if highlight_text:
        # Try to find and highlight part of answer
        areas = page.search_for(highlight_text[:30])
        for a in areas:
            page.add_highlight_annot(a)

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    path = f"page_{page_num}.png"
    pix.save(path)

    return path


# ================= METRICS =================
# Evaluate answer quality
def compute_metrics(query, answer, docs):
    try:
        q = embeddings.embed_query(query)
        a = embeddings.embed_query(answer)

        context = " ".join([d.page_content for d in docs])
        c = embeddings.embed_query(context)

        # Faithfulness
        faithfulness = cosine_similarity([a], [c])[0][0]

        # Answer relevance
        answer_relevance = cosine_similarity([q], [a])[0][0]

        # Context relevance
        doc_scores = [
            cosine_similarity([q], [embeddings.embed_query(d.page_content)])[0][0]
            for d in docs
        ]
        context_relevance = sum(doc_scores) / len(doc_scores)

        # Coverage
        coverage = min(1.0, context_relevance * 1.2)

        # Final accuracy score
        accuracy = (
            0.4 * faithfulness +
            0.3 * context_relevance +
            0.3 * answer_relevance
        )

        return {
            "faithfulness": round(faithfulness, 3),
            "context_relevance": round(context_relevance, 3),
            "answer_relevance": round(answer_relevance, 3),
            "context_coverage": round(coverage, 3),
            "accuracy": round(accuracy, 3)
        }

    except:
        return {}


# ================= GRAPH =================
# Plot metrics
def plot_metrics(metrics):
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6,4))
    plt.bar(names, values)
    plt.xticks(rotation=30)
    plt.ylim(0, 1)

    path = "metrics.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


# ================= RESOURCES =================
# Suggest papers and links based on query
def get_resources(query):
    q = query.lower()

    papers = []
    github = []

    if "rag" in q:
        papers.append("https://arxiv.org/abs/2005.11401")
        github.append("https://github.com/facebookresearch/rag")

    if "embedding" in q:
        papers.append("https://arxiv.org/abs/2212.10496")
        github.append("https://github.com/FlagOpen/FlagEmbedding")

    if not papers:
        papers.append("https://arxiv.org/abs/2005.11401")

    videos = [
        "https://www.youtube.com/watch?v=T-D1OfcDW1M",
        "https://www.youtube.com/watch?v=9AXP7tCI9PI"
    ]

    docs = [
        "https://python.langchain.com/docs/",
        "https://huggingface.co/docs"
    ]

    return papers, videos, github, docs


# Format links for UI display
def format_links(title, links):
    return f"### {title}\n" + "\n".join([f"- {l}" for l in links])


# ================= REPORT =================
# Save answer and metrics into file
def generate_report(answer, metrics):
    text = f"Answer:\n{answer}\n\nMetrics:\n{metrics}"
    file_path = "report.txt"

    with open(file_path, "w") as f:
        f.write(text)

    return file_path


# ================= MAIN RAG =================
def ask(query):
    docs = retriever.invoke(query)
    docs = docs[:3]

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer only from context.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt)

    metrics = compute_metrics(query, answer, docs)
    graph = plot_metrics(metrics)

    papers, videos, github, docs_links = get_resources(query)

    images = []
    info = []

    for d in docs:
        page = d.metadata.get("page", 0)
        img = render_page(page, answer)
        images.append((img, f"Page {page}"))

        info.append(d.page_content[:200])

    chat_history.append((query, answer))

    return (
        answer,
        images,
        info,
        metrics,
        graph,
        format_links("Research Papers", papers),
        format_links("YouTube Videos", videos),
        format_links("GitHub", github),
        format_links("Docs", docs_links),
        chat_history
    )


# ================= UI =================
with gr.Blocks() as demo:

    gr.Markdown("Grounded RAG Based LLM Assistant for Semiconductor Industry")

    query = gr.Textbox(label="Ask Question")
    answer_box = gr.Textbox(label="Answer")

    voice_btn = gr.Button("Voice Input")
    ask_btn = gr.Button("Ask")

    speak_btn = gr.Button("Read Answer")
    stop_btn = gr.Button("Stop")

    gallery = gr.Gallery()
    info = gr.JSON()
    metrics_box = gr.JSON()
    graph_output = gr.Image()

    papers_box = gr.Markdown()
    videos_box = gr.Markdown()
    github_box = gr.Markdown()
    docs_box = gr.Markdown()

    history_box = gr.JSON(label="Chat History")

    download_btn = gr.Button("Download Report")
    file_output = gr.File()

    # Ask query
    ask_btn.click(
        ask,
        inputs=query,
        outputs=[
            answer_box,
            gallery,
            info,
            metrics_box,
            graph_output,
            papers_box,
            videos_box,
            github_box,
            docs_box,
            history_box
        ]
    )

    # Voice input
    voice_btn.click(voice_input, outputs=query)

    # Speak answer
    def speak_wrapper(ans):
        speak_text(ans)

    speak_btn.click(speak_wrapper, inputs=answer_box, outputs=[])

    # Stop speaking
    stop_btn.click(stop_speaking)

    # Download report
    download_btn.click(
        generate_report,
        inputs=[answer_box, metrics_box],
        outputs=file_output
    )

demo.launch()