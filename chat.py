import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
embedd_model = os.getenv("EMBEDDING_MODEL")
llm_model = os.getenv("LANGUAGE_MODEL")

# Embedding model
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
    model=embedd_model, google_api_key=google_api_key
)

# Language model
LANGUAGE_MODEL = ChatGoogleGenerativeAI(
    model=llm_model, google_api_key=google_api_key
)

# Persistent directory for ChromaDB
CHROMA_PERSIST_DIR = "chromadb"

# Prompt template
PROMPT_TEMPLATE = """
You are an AI assistant for reading and understanding PDF documents.
Use the provided context to give a clear and helpful answer to the user's question.

If the answer can't be found in the document, respond with:
"Sorry, I couldn't find that information in the document."

Query: {user_query}
Context: {document_context}

Answer:
"""

# Storage path
PDF_STORAGE_PATH = "document_store/pdfs/"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Get a vector store for a specific PDF
def get_vector_store(collection_name):
    return Chroma(
        embedding_function=EMBEDDING_MODEL,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name
    )

# Save file locally
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Load and parse PDF using PDFPlumber
def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# Chunk the documents
def chunk_documents(raw_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_documents)

# Index documents
def index_documents(document_chunks, file_id):
    db = get_vector_store(file_id)
    index_flag_path = os.path.join(CHROMA_PERSIST_DIR, f"{file_id}.indexed")
    if os.path.exists(index_flag_path):
        return
    db.add_documents(document_chunks)
    db.persist()
    with open(index_flag_path, "w") as f:
        f.write("indexed")

# Search related chunks
def find_related_documents(query, file_id):
    db = get_vector_store(file_id)
    return db.similarity_search(query)

# Generate response
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL
    result = chain.invoke({
        "user_query": user_query,
        "document_context": context_text
    })
    return result.content

# Streamlit app
st.set_page_config(page_title="Chat with PDF")
st.title("📘 Chat with your PDF")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# Chat history session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

if uploaded_pdf:
    if uploaded_pdf.type != "application/pdf":
        st.error("❌ Please upload a PDF file")
    else:
        file_id = uploaded_pdf.name.replace(" ", "_").lower()
        saved_path = os.path.join(PDF_STORAGE_PATH, uploaded_pdf.name)

        # Reset history if file is different
        if file_id != st.session_state.last_file_id:
            st.session_state.chat_history = []
            st.session_state.last_file_id = file_id

        if not os.path.exists(saved_path):
            with st.spinner("📂 Saving file..."):
                saved_path = save_uploaded_file(uploaded_pdf)

        with st.spinner("⚙️ Processing document..."):
            documents = load_pdf_documents(saved_path)
            chunks = chunk_documents(documents)
            index_documents(chunks, file_id)

        st.success("✅ Document is ready. Ask your questions below.")

        # Show chat history
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

        # Chat input
        user_input = st.chat_input("Ask a question about the PDF...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("🤖 Thinking..."):
                related_docs = find_related_documents(user_input, file_id)
                response = generate_answer(user_input, related_docs)

            with st.chat_message("assistant"):
                st.markdown(response)

            # Store chat
            st.session_state.chat_history.append((user_input, response))
