import streamlit as st
import pdfplumber
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

import os  # For environment variable access

# Configure Generative AI with API Key from environment variables
genai_api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=genai_api_key)

# Configure Pinecone with API Key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "multilingual-e5-large"


index = pc.Index('multilingual-e5-large')

# Load the sentence transformer model for embedding
embedder = SentenceTransformer('all-mpnet-base-v2')

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to store document embeddings in Pinecone
def store_document_in_pinecone(doc_text):
    # Split text into smaller chunks for embedding
    sentences = doc_text.split('. ')
    embeddings = embedder.encode(sentences)
    
    # Store in Pinecone
    for i, emb in enumerate(embeddings):
        index.upsert([(f"sentence-{i}", emb, {'text': sentences[i]})])

# Function to retrieve relevant chunks from Pinecone
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    relevant_chunks = [match['metadata']['text'] for match in results['matches']]
    return " ".join(relevant_chunks)


# Function to get a summary from Gemini-pro API
def summarize_text(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Please summarize the following text:\n\n{text}"])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Function to answer a question using the retrieved text and Gemini-pro
def question_text(retrieved_text, question):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Answer the following question based on the provided text:\n\nText: {retrieved_text}\n\nQuestion: {question}"])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
def main():
    st.title("RAG-based PDF QA Bot with Gemini-pro")
    
    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        
        # Store the document embeddings in Pinecone
        store_document_in_pinecone(text)
        
        # Display extracted text (first 500 characters)
        display_text = text[:500] + ('...' if len(text) > 500 else '')
        st.subheader("Extracted Text")
        st.text_area("Text from PDF", display_text, height=300)

        # Get a summary
        if st.button("Get Summary"):
            summary = summarize_text(text)
            st.subheader("Summary")
            st.write(summary)

        # Ask a question
        question = st.text_input("Enter your question about the text")
        if st.button("Get Answer"):
            if question:
                # Retrieve relevant chunks from Pinecone
                relevant_text = retrieve_relevant_chunks(question)
                
                # Get the answer from Gemini-pro
                answer = question_text(relevant_text, question)
                
                st.subheader("Retrieved Text")
                st.write(relevant_text)
                
                st.subheader("Answer")
                st.write(answer)
            else:
                st.warning("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
