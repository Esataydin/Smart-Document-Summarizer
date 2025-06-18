# 📄 DocIQ – Smart Document Summarizer & Q&A (Local App)

**DocIQ** is a local, privacy-friendly document analysis tool built with **Streamlit**. It allows users to upload a PDF, extract and summarize its content based on audience role (CEO, Engineer, Legal Analyst, etc.), and ask natural language questions with precise, context-aware answers.

---

## ✨ Features

### 📌 Role-Based Summarization
- Generate tailored summaries for different audiences:
  - CEO: High-level impact and business takeaways
  - Engineer: Technical metrics and architecture
  - Legal Analyst: Compliance, dates, legal terms
- Uses `google/pegasus-xsum` for high-quality summarization.

### 🔍 Intelligent Q&A
- Ask natural language questions based on the document.
- Finds the most relevant excerpt using semantic search (`bge-large-en + FAISS`).
- Answers are generated using a real QA model (`roberta-base-squad2`).
- Displays confidence score for transparency.

### 🧾 Export
- Download summary or full Q&A output as `.txt` files.

---

## 🛠 Tech Stack

- **Streamlit** – Web UI
- **Sentence-Transformers** – `BAAI/bge-large-en` for semantic embeddings
- **FAISS** – Fast vector similarity search
- **Transformers (Hugging Face)**:
  - `google/pegasus-xsum` – Summarization
  - `deepset/roberta-base-squad2` – Question Answering
- **PyMuPDF** – PDF text extraction

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Esataydin/Smart-Document-Summarizer.git
   cd Smart-Document-Summarizer
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run main.py
   ```

* Upload a PDF document.
* Alternatively, try the included test.pdf file in the repository/folder for a quick demo.
* Choose a summary role and generate a tailored summary.
* Ask questions about the document and get relevant answers with context.
* Download summaries and Q\&A results for offline use.

---

## 📂 Folder Structure
├── main.py              # Streamlit application <br>
├── requirements.txt     # Dependencies <br>
├── test.pdf             # Example document for test <br>
├── test.txt             # Example document for test(future work) <br>
└── README.md            # Project documentation <br>




---

## 📌 Notes
- All models run locally — no OpenAI API key required
- Works offline and respects document privacy
- Best performance with medium-length PDFs (~5–10 pages)

---

## How It Works

1. **Document Processing:**

   * The uploaded PDF is parsed, and the text is extracted.
   * The text is chunked into overlapping segments to maintain context.

2. **Embedding & Indexing:**

   * Each text chunk is converted into embeddings using a pre-trained SentenceTransformer model.
   * The embeddings are indexed using FAISS for fast similarity searches.

3. **Summarization:**

   * Users select a role, which defines the style and focus of the summary.
   * The summarization model generates summaries of the first few chunks, tailored to the selected role.

4. **Q\&A Interface:**

   * User questions are embedded and matched against the document chunks.
   * Top relevant chunks are retrieved and combined with the question to generate an answer via the summarization pipeline.

---

## Role Prompts

The app customizes the summaries using role-specific prompts:

* **General:** Clear and concise summary.
* **CEO:** Business value and high-level takeaways without jargon.
* **Engineer:** Technical details, architecture, and metrics.
* **Legal Analyst:** Legal terms, obligations, dates, and compliance.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

---

## Acknowledgments

* Thanks to Hugging Face and SentenceTransformers for providing excellent NLP models.
* FAISS by Facebook AI for scalable vector search.
* Streamlit for easy app development.

---

If you have any questions or suggestions, feel free to reach out!

---

*Created by Esataydin*
