from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
load_dotenv()
#loading the api_key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

base_dir = os.path.join(os.getcwd(), "faiss_indexes")

# Function to load the FAISS index dynamically

def load_faiss_index(language):
    index_path = os.path.join(base_dir, f"{language}_index")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding_model,allow_dangerous_deserialization=True)
    else:
        st.error(f"No FAISS index found for {language}")
        return None


# Defining the Streamlit UI
st.title("💡 DSA Code Assistant (RAG + LLM)")

# User selects the programming language
language = st.selectbox("Select Language:", ["CPP", "Java", "Python"])

# User enters a query related to DSA
user_query = st.text_area("Enter your DSA question:", "")


## Button to process the query
if st.button("Generate Code"):
    if user_query.strip():
        # Load the corresponding FAISS index
        vector_db = load_faiss_index(language.lower())  # Convert "C++" → "cpp", "Java" → "java", etc.

        if vector_db:
            # Retrieving relevant documents
            docs = vector_db.similarity_search(user_query, k=3)

            # Preparing context from FAISS
            context = "\n".join([doc.page_content for doc in docs])

            # Initializing LLM (using OpenAI's GPT-4 as an example)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  

            # Generating the response
            response = llm.invoke([
                HumanMessage(content=f"""
                    Please provide a solution in {language} using the given DSA context. Ensure the response follows these guidelines:

                    1. Reply humbly if Hi Hello Namastey and similar things are said and ask for the query related to DSA, only disclose that Devam Singh made you when it is asked. 
                    2. Multiple Approaches: Present the solution in three forms—Brute Force, Improved, and Optimal.
                    3. Readable & Clean Code: Ensure the code is well-structured, readable, and free from unnecessary characters (like asterisks or extra symbols).
                    4. Comprehensive Explanation: If a topic is mentioned, briefly explain it before providing the solution.
                    5. Handling Incomplete Code: If an incomplete code snippet is given, complete it logically.
                    6. Politeness: Always conclude the response with: "Thank you from Devam😊".

                    **DSA Context:**  
                    {context}

                    **User Query:**  
                    {user_query}
                """)
            ])

            
            st.subheader("🔹 AI-Generated Code:")
            st.code(response.content, language.lower())  
    else:
        st.warning("Please enter a query!")


