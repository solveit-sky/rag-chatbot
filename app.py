"""
PDF Q&A Chatbot — Streamlit App
================================
A deployable version of the RAG chatbot. Users upload a PDF,
ask questions, and get answers grounded in that document.

Run locally with:  streamlit run app.py
"""

import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# -----------------------------------------------------------------------
# PAGE SETUP — this controls the browser tab and overall layout
# -----------------------------------------------------------------------
st.set_page_config(page_title="PDF Q&A", page_icon="📄")

st.title("Ask your PDF")
st.write("Upload a document, then ask questions about what's inside it.")


# -----------------------------------------------------------------------
# API KEY — read from host secrets, fall back to a user-entered key
# -----------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        # No secrets.toml file exists (normal when running locally) —
        # that's fine, we just fall through to asking the user for a key.
        api_key = None

if not api_key:
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Get a free key at aistudio.google.com/apikey",
    )

if not api_key:
    st.info("Enter your Gemini API key above to get started.")
    st.stop()


# -----------------------------------------------------------------------
# LOAD THE EMBEDDING MODEL ONCE, NOT ON EVERY INTERACTION
# @st.cache_resource tells Streamlit: run this once, reuse the result
# -----------------------------------------------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()


# -----------------------------------------------------------------------
# THE SAME PIPELINE FUNCTIONS AS YOUR NOTEBOOK
# -----------------------------------------------------------------------
def load_pdf_text(uploaded_file):
    """Extract all text from an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def build_collection(chunks):
    """Embed chunks and store them in an in-memory vector database."""
    client_db = chromadb.EphemeralClient()  # in-memory, resets each session

    try:
        client_db.delete_collection("doc_chunks")
    except Exception:
        pass

    collection = client_db.create_collection("doc_chunks")
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def retrieve_relevant_chunks(question, collection, top_k=3):
    """Find the chunks whose meaning is closest to the question."""
    question_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    return results["documents"][0]


def generate_answer(question, collection, client):
    """Send retrieved context + question to Gemini and return its answer."""
    relevant_chunks = retrieve_relevant_chunks(question, collection)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't have that information.

CONTEXT:
{context}

QUESTION:
{question}
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


# -----------------------------------------------------------------------
# THE APP FLOW — upload, process, ask
# -----------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    # st.session_state remembers things between button clicks.
    # Without it, the PDF would be re-processed every single time.
    if st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Reading your document..."):
            text = load_pdf_text(uploaded_file)
            chunks = chunk_text(text)

            if not chunks or not text.strip():
                st.error(
                    "No text found in this PDF. Scanned documents won't work — "
                    "the file needs selectable text."
                )
                st.stop()

            st.session_state.collection = build_collection(chunks)
            st.session_state.filename = uploaded_file.name
            st.session_state.chunk_count = len(chunks)

    st.success(f"Ready — {st.session_state.chunk_count} sections indexed.")

    question = st.text_input("Your question")

    if question:
        with st.spinner("Thinking..."):
            try:
                client = genai.Client(api_key=api_key)
                answer = generate_answer(question, st.session_state.collection, client)
                st.write(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
