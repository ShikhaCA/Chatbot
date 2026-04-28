import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------------
# STEP 1: Load PDF
# -------------------------------
pdf_path ="RAG Architectures.pdf"   

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# -------------------------------
# STEP 2: Split into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)
print(f"Total chunks: {len(docs)}")

# -------------------------------
# STEP 3: Create embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# STEP 4: Store in FAISS
# -------------------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# -------------------------------
# STEP 5: Load LLM (local model)
# -------------------------------
model_name = "distilgpt2"   # small + runs on CPU

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# STEP 6: Chat loop
# -------------------------------
print("\n📚 PDF Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask your question: ")

    if query.lower() == "exit":
        break

    # Step 6.1: Retrieve relevant chunks
    results = vectorstore.similarity_search(query, k=3)

    # Step 6.2: Combine context
    context = "\n\n".join([doc.page_content for doc in results])

    # Step 6.3: Create prompt
    prompt = f"""
You are a helpful assistant. Answer ONLY from the given context.

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 6.4: Generate answer
    
    response = llm.invoke(prompt)

    print("\n🤖 Answer:\n")
    print(response)

 