import streamlit as st
import tempfile
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Smart PDF Chatbot (Report Analyzer)")

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

# -------------------------------
# SMART ANSWER FUNCTION
# -------------------------------
def analyze_report(context, question):
    if not context:
        return "I don't know"

    question = question.lower()
    lines = context.split("\n")

    findings = []

    # Detect medical abnormal values
    for line in lines:
        l = line.lower()

        if any(word in l for word in ["low", "high", "below", "above"]):
            findings.append(line.strip())

        # detect numeric abnormal patterns
        if re.search(r"\b(low|high|below|above)\b", l):
            findings.append(line.strip())

    if "low" in question or "high" in question:
        if findings:
            return "🔍 Abnormal Findings:\n\n" + "\n".join(set(findings))
        else:
            return "No abnormal (low/high) values found."

    if "summary" in question:
        return context[:500]

    # default fallback
    for line in lines:
        if len(line.strip()) > 50:
            return line.strip()

    return context[:300]

# -------------------------------
# CHAT INPUT
# -------------------------------
user_input = st.chat_input("Ask about your report (e.g., 'what is low?', 'summary')")

if user_input:

    st.session_state.messages.append(("user", user_input))

    if st.session_state.vectorstore is None:
        response = "⚠️ Please upload a PDF first."
    else:
        with st.spinner("Analyzing..."):

            results = st.session_state.vectorstore.similarity_search(user_input, k=4)

            context = ""
            for doc in results:
                context += doc.page_content + "\n\n"

            response = analyze_report(context, user_input)

            # Add source info
            response += "\n\n📌 Source: Uploaded PDF"

    st.session_state.messages.append(("assistant", response))

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)