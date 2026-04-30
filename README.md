Smart PDF Chatbot (RAG-Based)

This project is a simple implementation of a PDF-based chatbot built using a Retrieval-Augmented Generation (RAG) approach. The application allows users to upload one or more PDF documents and interact with them through a question-answering interface. The goal of this project is to explore how document data can be processed and retrieved efficiently to provide relevant responses based on user queries.

The system works by first extracting text from uploaded PDF files and then splitting that text into smaller, manageable chunks. These chunks are converted into vector embeddings using a Hugging Face model, which helps represent the semantic meaning of the text. The embeddings are stored in a FAISS vector database, allowing the system to perform fast and efficient similarity-based searches.

When a user asks a question, the application converts the query into an embedding and searches for the most relevant chunks in the database. The retrieved context is then used to generate a response, which is displayed through an interactive chat interface built with Streamlit.

Features
1-Upload and process single or multiple PDF files
2-Ask questions related to the uploaded documents
3-Retrieve relevant information using semantic search
4-Display responses in a chat-based interface
5-View retrieved context for better understanding of results
6-Clear chat functionality for better user interaction

Technologies Used
Python
Streamlit for building the user interface
LangChain for document processing and pipeline management
FAISS for vector storage and similarity search
Hugging Face Embeddings (all-MiniLM-L6-v2) for semantic representation
