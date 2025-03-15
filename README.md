# AI_DSA_ASSISTANT
AI DSA Assistant to solve coding related problems of JAVA, C++ and Python. You can not only ask about the incomplete code but every problem will be provided with 3 approaches Brute Force, Better and Optimal. It can describe the hardest of hard topics and problem considering it is RAG based ChatBot having several DSA books as external data source.
<img width="958" alt="image" src="https://github.com/user-attachments/assets/771c46a9-6252-466e-bf6d-a124856b2200" />


Deployed link : https://usingllm.streamlit.app/

## Concept
This system leverages three DSA books—DSA in Java, Python, and C++—as external sources, storing their embeddings in separate FAISS indexes using Meta’s FAISS as the vector database. The embeddings are generated using Hugging Face’s SentenceTransformer, while Gemini 1.5 Pro serves as the LLM, integrated through ChatGoogleGenerativeAI.

To enhance efficiency, the system incorporates both LLM-driven retrieval and a decision-based approach, allowing users to specify whether they want a solution in Python, Java, or C++. This hybrid method ensures faster, more precise retrieval by directly accessing the relevant FAISS index instead of relying solely on the LLM to search across the entire database.

## Usability
Designed for daily coding tasks, this system is particularly useful for DSA-related problem-solving. Key features include:
✅ Code Completion – Automatically completes any incomplete code snippets.
✅ Concept Explanation – Simplifies complex DSA topics with clear explanations and examples.
✅ Multi-Level Solutions – Provides three variations for any problem: Brute Force, Improved, and Optimal solutions.

This intelligent retrieval system significantly enhances learning, debugging, and competitive coding by making problem-solving faster, structured, and language-specific. 

## Credits
-Devam Singh
-IIIT Ranchi (B.Tech CSE)
