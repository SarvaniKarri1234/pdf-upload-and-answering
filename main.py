import streamlit as st
import pdfplumber
import pickle
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define function to extract text from the PDF
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        st.error(f"Failed to read PDF file: {str(e)}")
    return text

# Define function to process queries
def process_query(query, VectorStore, qa_pipeline):
    docs = VectorStore.similarity_search(query=query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Set up question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

# Streamlit UI
st.title("Interactive PDF Query System")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    pdf_text = extract_pdf_text(uploaded_file)
    st.write("Text extracted from PDF...")

    # Split the extracted text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=pdf_text)
    st.write(f"Number of chunks: {len(chunks)}")

    # Embed the chunks using HuggingFace SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    # Save embeddings to disk (optional, for reusability)
    store_name = uploaded_file.name[:-4]  # Remove file extension from the name
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    # User query input field
    query = st.text_input("Ask a question about the PDF:")

    if query:
        # Process the query and return the answer
        answer = process_query(query, VectorStore, qa_pipeline)
        st.write(f"Answer: {answer}")
else:
    st.write("Please upload a PDF file to get started.")
