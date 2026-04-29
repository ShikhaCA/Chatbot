import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Smart PDF Chatbot (General Purpose)")

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

# -------------------------------
# PROCESS PDF
# -------------------------------
if uploaded_files:

    all_docs = []

    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

    st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) processed!")

    st.info("💡 Ask anything about your PDF")

# -------------------------------
# GENERIC ANSWER FUNCTION
# -------------------------------
def generate_answer(context, question):
    if not context:
        return "I don't know."

    # Split into sentences/lines
    lines = context.split("\n")

    # Try to find most relevant sentence
    question_words = question.lower().split()

    best_line = ""
    max_score = 0

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for word in question_words if word in line_lower)

        if score > max_score and len(line.strip()) > 40:
            max_score = score
            best_line = line.strip()

    if best_line:
        return best_line

    # fallback
    return context[:400]

# -------------------------------
# QUESTION SUGGESTIONS
# -------------------------------
st.markdown("### 💬 Try asking:")
st.markdown("""
- What is this document about?
- Give me summary
- Explain this topic
- Key points from document
""")

# -------------------------------
# CHAT INPUT
# -------------------------------
user_input = st.chat_input("Ask anything about your PDF...")

if user_input:

    st.session_state.messages.append(("user", user_input))

    if st.session_state.vectorstore is None:
        response = "⚠️ Please upload a PDF first."
    else:
        with st.spinner("Searching..."):

            results = st.session_state.vectorstore.similarity_search(user_input, k=4)

            context = ""
            sources = []

            for doc in results:
                context += doc.page_content + "\n\n"
                sources.append(doc.metadata.get("source", "Uploaded PDF"))

            if not context.strip():
                response = "⚠️ No relevant information found."
            else:
                response = generate_answer(context, user_input)

            # Add sources
            response += "\n\n📌 Source: Uploaded PDF"

            # Debug view
            with st.expander("🔍 Retrieved Context"):
                st.write(context[:800])

    st.session_state.messages.append(("assistant", response))

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)