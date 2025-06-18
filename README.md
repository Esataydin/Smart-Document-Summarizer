# DocIQ - Smart Document Summarizer + Search

DocIQ is a Streamlit-based web application designed to help users upload PDF documents and perform intelligent summarization and semantic search on the content. It leverages powerful NLP models to provide role-based summaries and an interactive Q&A interface that finds relevant excerpts in the document and generates precise answers.

---

## Features

- **PDF Upload & Text Extraction:** Extracts and processes text from uploaded PDF files.
- **Text Chunking:** Splits large documents into manageable overlapping chunks for better embeddings.
- **Semantic Search:** Uses FAISS for fast similarity search on vector embeddings of document chunks.
- **Role-Based Summarization:** Generates summaries tailored to different roles such as General, CEO, Engineer, and Legal Analyst.
- **Interactive Q&A:** Ask questions about the document and get context-aware answers with relevant excerpts.
- **Downloadable Results:** Allows downloading generated summaries and full Q&A results as text files.

---

## Technologies Used

- **Streamlit** - Web application framework for Python.
- **PyMuPDF (fitz)** - PDF parsing and text extraction.
- **Sentence-Transformers** - Producing vector embeddings for semantic search.
- **FAISS** - Efficient similarity search library by Facebook AI.
- **Transformers (Hugging Face)** - State-of-the-art summarization pipeline using `facebook/bart-large-cnn`.
- **Base64** - For generating downloadable summary and Q&A files.

---

## Installation

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
   
---

## Usage

Run the Streamlit app locally:

```bash
streamlit run main.py
```

* Upload a PDF document.
* Alternatively, try the included test.pdf file in the repository/folder for a quick demo.
* Choose a summary role and generate a tailored summary.
* Ask questions about the document and get relevant answers with context.
* Download summaries and Q\&A results for offline use.

---

## Notes

- The current summarization model (`facebook/bart-large-cnn`) used locally is relatively small and may not provide perfect results for very large or complex documents.
- Performance and summary quality will improve once a larger or more advanced model is integrated in future updates.

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
