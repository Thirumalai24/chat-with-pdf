
# 📄 Chat with your PDF (LangChain + Gemini + ChromaDB)

A Streamlit web app that allows you to upload a PDF and chat with it using Google's Gemini Pro (1.5) model. It intelligently answers your questions based on the contents of the uploaded document using semantic search and large language models.

---

## 🚀 Features

- Upload and interact with any PDF document
- Uses **Google Gemini Pro** for natural language answers
- Semantic search via **ChromaDB** and **vector embeddings**
- Persistent vector storage to avoid re-processing files
- Clean chat interface with history support
- Validates PDF file type before processing

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** – Frontend interface
- **LangChain** – LLM orchestration and chaining
- **Google Gemini Pro (1.5)** – LLM for answering questions
- **Google Generative AI Embeddings** – For vectorization
- **ChromaDB** – Local vector store
- **PDFPlumber** – For parsing PDF content

---

## 📦 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Thirumalai24/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API Key:**

   - Create a `.env` file in the project root:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```

---

## 🧪 Usage

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **In the browser:**
   - Upload a PDF file.
   - Ask any questions related to the content.
   - The model will respond based on what's inside the document.

---

## 📁 Project Structure

```
chat-with-pdf-gemini/
├── app.py                     # Main Streamlit application
├── .env                       # API key configuration
├── requirements.txt           # Dependencies
├── chromadb/                  # ChromaDB persistent storage
├── document_store/
│   └── pdfs/                  # Uploaded PDF files
```

---

## ⚠️ Notes

- Only `.pdf` files are allowed.
- Already indexed PDFs won't be reprocessed to improve performance.
- Make sure your Google API key has access to both **Gemini Pro** and **Embedding API**.

---

## 📌 Example Questions

- What is K-means clustering?
- Summarize the introduction.
- Explain with examples from the document.
- What are the key takeaways from Chapter 3?

---

## 📃 License

This project is open source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
```

