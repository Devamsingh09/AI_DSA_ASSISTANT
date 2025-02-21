import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
pdf_files = {
    "cpp": "data/pdfs/dsa_cpp.pdf",
    "java": "data/pdfs/dsa_java.pdf",
    "python": "data/pdfs/dsa_python.pdf",
}

save_dir = "faiss_indexes"  # Directory to save indexes

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load the embedding model once
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to process and save FAISS index
def process_and_save_faiss(language, pdf_path):
    print(f"Processing {language}...")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)

    vector_db = FAISS.from_documents(final_docs, embedding)
    index_path = os.path.join(save_dir, f"{language}_index")

    vector_db.save_local(index_path)
    print(f"{language.capitalize()} index saved successfully at {index_path}")

# Process and save indexes for each language
for lang, path in pdf_files.items():
    process_and_save_faiss(lang, path)

print("All indexes saved successfully! 🎯")
