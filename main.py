import streamlit as st
import fitz  # PyMuPDF for PDF reading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import pipeline as hf_pipeline
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 300))
OVERLAP_SIZE = int(os.getenv('OVERLAP_SIZE', 50))
MAX_DOCUMENT_SIZE = int(os.getenv('MAX_DOCUMENT_SIZE', 15000))

# === Load Models ===
st.set_page_config(page_title="DocIQ - Local", layout="wide")
st.title("📄 DocIQ - Smart Document Summarizer & Q&A (Full Doc Summary)")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('BAAI/bge-large-en')  # High-accuracy embedding model
    summarizer = pipeline("summarization", model="google/pegasus-xsum")  # Full-document summary
    qa_model = hf_pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embed_model, summarizer, qa_model

embed_model, summarizer, qa_pipeline = load_models()

# === PDF Upload ===
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=None, overlap=None):
    if max_tokens is None:
        max_tokens = CHUNK_SIZE
    if overlap is None:
        overlap = OVERLAP_SIZE
    
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

def truncate_text(text, max_words=300):
    words = text.split()
    return " ".join(words[:max_words])

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors, texts):
        self.index.add(np.array(vectors).astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vector, k=3):
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k)
        return [self.texts[i] for i in I[0]]

def create_download_button(content, filename, label):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    return href

role_prompts = {
    "General": "Summarize the following text clearly and concisely:",
    "CEO": "You are summarizing this text for a busy CEO. Focus on high-level takeaways, business value, and outcomes. Avoid technical jargon:",
    "Engineer": "Summarize this text for an engineer. Highlight system architecture, technologies used, performance metrics, and technical details:",
    "Legal Analyst": "Summarize this text for a legal analyst. Emphasize obligations, dates, legal terms, and compliance risks:",
}

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_text_from_pdf(uploaded_file)
        if len(text.split()) > MAX_DOCUMENT_SIZE:
            st.warning("⚠️ Document is very long. Consider uploading a shorter file or using fewer pages.")

        chunks = chunk_text(text, max_tokens=CHUNK_SIZE, overlap=OVERLAP_SIZE)
        embeddings = embed_model.encode(chunks, convert_to_numpy=True)

        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, chunks)
        st.success("✅ Document processed.")

        # === Summarization with Role Selection ===
        st.subheader("📝 Generate Summary")
        role = st.selectbox("Select role for summary:", list(role_prompts.keys()))
        summary_text = ""
        if st.button("Generate Summary"):
            st.info(f"Document split into {len(chunks)} parts for summarization.")
            summaries = []
            for chunk in chunks[:5]:
                safe_chunk = truncate_text(chunk)
                input_text = role_prompts[role] + " " + safe_chunk
                try:
                    result = summarizer(input_text, max_length=120, min_length=50, do_sample=False)
                    chunk_summary = result[0]['summary_text']
                    st.markdown(f"✅ **Chunk Summary:** {chunk_summary}")
                    summaries.append(chunk_summary)
                except Exception as e:
                    st.error(f"❌ Failed to summarize a chunk: {e}")

            if summaries:
                combined = " ".join(summaries)
                try:
                    final_result = summarizer(combined, max_length=200, min_length=80, do_sample=False)
                    summary_text = final_result[0]['summary_text']
                except Exception as e:
                    st.warning("⚠️ Could not re-summarize. Displaying combined chunk summaries.")
                    summary_text = combined

                st.subheader(f"📌 Summary for {role}:")
                st.write(summary_text)

                download_link = create_download_button(summary_text, f"summary_{role}.txt", "📥 Download Summary")
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.warning("❌ Could not generate any summaries.")

        # === Q&A Interface ===
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Searching relevant context..."):
                q_vec = embed_model.encode(["Represent this question for retrieving relevant documents: " + query], convert_to_numpy=True)[0]
                top_chunks = store.search(q_vec, k=3)
                context = " ".join(top_chunks)

                st.subheader("🧠 Answer:")
                try:
                    answer = qa_pipeline({'question': query, 'context': context})
                    st.write(f"**{answer['answer']}** (confidence: {round(answer.get('score', 0)*100, 2)}%)")

                    full_qa_output = f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:\n{answer['answer']}"
                    qa_link = create_download_button(full_qa_output, "qa_result.txt", "📥 Download Full Q&A")
                    st.markdown(qa_link, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
