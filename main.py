


import streamlit as st
import fitz  # PyMuPDF for PDF reading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import base64

# === Load Models ===
st.set_page_config(page_title="DocIQ - Local", layout="wide")
st.title("📄 DocIQ - Smart Document Summarizer + Search")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embed_model, summarizer

embed_model, summarizer = load_models()

# === PDF Upload ===
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=150, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

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
        if len(text.split()) > 8000:
            st.warning("⚠️ Document is very long. Consider uploading a shorter file for better results.")

        chunks = chunk_text(text)
        embeddings = embed_model.encode(chunks, convert_to_numpy=True)

        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, chunks)
        st.success("✅ Document processed.")

        # === Summarization with Role Selection ===
        st.subheader("📝 Generate Summary")
        role = st.selectbox("Select role for summary:", list(role_prompts.keys()))
        summary_text = ""
        if st.button("Generate Summary"):
            for chunk in chunks[:3]:
                result = summarizer(role_prompts[role] + " " + chunk, max_length=150, min_length=40, do_sample=False)
                summary_text += "\n- " + result[0]['summary_text']
            st.subheader(f"📌 Summary for {role}:")
            st.write(summary_text)

            download_link = create_download_button(summary_text, f"summary_{role}.txt", "📥 Download Summary")
            st.markdown(download_link, unsafe_allow_html=True)

        # === Q&A Interface ===
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Searching relevant context..."):
                q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
                top_chunks = store.search(q_vec, k=3)
                st.subheader("🔍 Most Relevant Excerpts:")
                snippet_text = ""
                for i, chunk in enumerate(top_chunks):
                    excerpt = f"**{i+1}.** {chunk[:500]}..."
                    st.markdown(excerpt)
                    snippet_text += excerpt + "\n\n"

                combined_context = " ".join(top_chunks[:2])
                result = summarizer(combined_context + " Question: " + query, max_length=150, min_length=40, do_sample=False)
                st.subheader("🧠 Answer:")
                st.write(result[0]['summary_text'])

                full_qa_output = f"Question:\n{query}\n\nRelevant Excerpts:\n{snippet_text}\n\nAnswer:\n{result[0]['summary_text']}"
                qa_link = create_download_button(full_qa_output, "qa_result.txt", "📥 Download Full Q&A")
                st.markdown(qa_link, unsafe_allow_html=True)

