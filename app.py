import streamlit as st
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pdf2image import convert_from_path
import pytesseract
import re
import spacy
import sys


nlp = spacy.load("en_core_web_sm")
sys.modules["torch.classes"] = None







# --- Streamlit config ---
st.set_page_config(page_title="PDF Chatbot with LLaMA (LangChain Memory)", layout="wide")

# --- Initialize LLM and embeddings ---
llm = OllamaLLM(model="llama3", base_url="http://ollama:11434")
embedder = OllamaEmbeddings(model="llama3")

# --- Enhanced Session state with LangChain memory ---
if "memory" not in st.session_state:
    st.session_state.memory = InMemoryChatMessageHistory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {}
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# --- Intent templates ---
INTENT_TEMPLATES = {
    "list_files": [
        "what files have I uploaded",
        "list uploaded files", 
        "show me my files",
        "which pdfs are uploaded"
    ],
    "ask_pdf": [
        "tell me about the pdf report",
        "what does the document say",
        "details from the uploaded file",
        "information in the pdf",
        "explain what's in the document",
        "summarize the file"
    ],
    "general_chat": [
        "hello",
        "hi", 
        "how are you",
        "what is",
        "who is",
        "where is",
        "when did",
        "how does",
        "why is",
        "tell me about",
        "explain",
        "i want to know about",
        "give me information on",
        "define",
        "describe",
        "can we play"
    ]
}

# Precompute embeddings for intent templates once
if "intent_embeddings" not in st.session_state:
    st.session_state.intent_embeddings = {
        intent: [embedder.embed_query(text) for text in samples]
        for intent, samples in INTENT_TEMPLATES.items()
    }

# --- LangChain Memory Setup ---
def get_memory():
    return st.session_state.memory

# Chat prompt for general conversation
general_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You have access to conversation history and can provide contextual responses.Answer clearly without assumptions"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# PDF-specific prompt
pdf_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant answering questions **only** using the provided PDF context. "
     "If the answer is not in the context, respond with: 'I could not find the answer in the document.' "
     "Do not use any outside knowledge. Do not guess."
     "If the question asks you to explain in general way, use your knowledge to answer"
     "Do not hallucinate"
     ),
    MessagesPlaceholder("history"),
    ("human", "Context from PDF:\n\n{pdf_context}\n\nQuestion: {input}")
])

def build_general_chain():
    return RunnableWithMessageHistory(
        general_chat_prompt | llm,
        lambda session_id: get_memory(),
        input_messages_key="input",
        history_messages_key="history"
    )

def build_pdf_chain():
    return RunnableWithMessageHistory(
        pdf_chat_prompt | llm,
        lambda session_id: get_memory(),
        input_messages_key="input", 
        history_messages_key="history"
    )

general_chain = build_general_chain()
pdf_chain = build_pdf_chain()

# --- Utility functions ---


def query_words_in_pdf(query, pdf_text):
    doc = nlp(query)
    content_words = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"} and len(token.text) > 3]
    for word in content_words:
        if word in pdf_text.lower():
            return True
    return False


def ocr_pdf_to_text(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        page_text = pytesseract.image_to_string(page)
        text += page_text + "\n"
    return text

def embed_pdf_if_needed(pdf_data):
    if pdf_data["vectorstore"] is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        docs = splitter.create_documents([pdf_data["text"]])
        pdf_data["vectorstore"] = FAISS.from_documents(
            documents=docs,
            embedding=embedder
        )


def cosine_similarity(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent(query, threshold=0.65):
    query_emb = embedder.embed_query(query.lower())
    best_intent = None
    best_score = -1

    for intent, embeddings in st.session_state.intent_embeddings.items():
        for emb in embeddings:
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score < threshold:
        return "general_chat"
    return best_intent

# --- Enhanced handlers with LangChain memory ---
def handle_list_files():
    files = list(st.session_state.pdf_texts.keys())
    if not files:
        return "No files uploaded yet."
    return "Uploaded files:\n" + "\n".join(files)

def handle_general_chat(query):
    memory = get_memory()
    memory.add_user_message(query)
    response = general_chain.invoke(
        {"input": query},
        {"configurable": {"session_id": st.session_state.session_id}}
    )
    memory.add_ai_message(response)
    return str(response)

def handle_ask_pdf(query, fname):
    pdf_data = st.session_state.pdf_texts[fname]
    embed_pdf_if_needed(pdf_data)
    retriever = pdf_data["vectorstore"].as_retriever()
    relevant_docs = retriever.invoke(query)
    pdf_context = "\n".join([doc.page_content for doc in relevant_docs[:5]])
    memory = get_memory()
    memory.add_user_message(query)
    response = pdf_chain.invoke(
        {"input": query, "pdf_context": pdf_context},
        {"configurable": {"session_id": st.session_state.session_id}}
    )
    memory.add_ai_message(str(response))
    return str(response)


def get_best_matching_pdf(query):
    query_emb = embedder.embed_query(query.lower())
    best_pdf = None
    best_score = -1

    for fname, pdf_data in st.session_state.pdf_texts.items():
        if pdf_data["vectorstore"] is None:
            embed_pdf_if_needed(pdf_data)

        retriever = pdf_data["vectorstore"].as_retriever()
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs[:3]])
        context_emb = embedder.embed_query(context)

        score = cosine_similarity(query_emb, context_emb)
        if score > best_score:
            best_score = score
            best_pdf = fname

    return best_pdf if best_score > 0.5 else None


def answer_query(query):
    intent = detect_intent(query)
    st.caption(f"🧭 Detected intent: `{intent}`")

    matched_pdf = st.session_state.active_pdf


    if not matched_pdf:
        query_lower = query.lower()
        for fname, pdf_data in st.session_state.pdf_texts.items():
            if query_words_in_pdf(query, pdf_data["text"]):
                matched_pdf = fname
                st.caption(f"📄 Found related content in `{fname}` → using PDF RAG.")
                break


    if not matched_pdf:
        matched_pdf = get_best_matching_pdf(query)
        if matched_pdf:
            st.caption(f"📄 Best semantic match found: `{matched_pdf}` → using PDF RAG.")


    if matched_pdf:
        return handle_ask_pdf(query, matched_pdf)


    return handle_general_chat(query)


def clear_session_memory():
    st.session_state.memory = InMemoryChatMessageHistory()
    st.session_state.chat_history = []
    st.session_state.pdf_texts = {}
    st.session_state.active_pdf = None

def clear_chat_only():
    st.session_state.memory = InMemoryChatMessageHistory()
    st.session_state.chat_history = []

def get_memory_stats():
    messages = st.session_state.memory.messages
    return {
        "total_messages": len(messages),
        "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
        "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
        "pdfs": len(st.session_state.pdf_texts)
    }

if st.sidebar.button("🗑️ Clear All Memory"):
    clear_session_memory()
    st.sidebar.success("All memory cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("📁 PDF Manager")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    filename = uploaded_file.name
    if filename not in st.session_state.pdf_texts:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        if len(text.strip()) < 100:
            text = ocr_pdf_to_text(tmp_path)
        st.session_state.pdf_texts[filename] = {
            "text": text,
            "path": tmp_path,
            "vectorstore": None
        }
        st.sidebar.success(f"Uploaded: {filename}")

if st.session_state.pdf_texts:
    st.sidebar.markdown("### 📄 Loaded PDFs")
    for fname in st.session_state.pdf_texts:
        if st.sidebar.button(f"View {fname}", key=f"btn_{fname}"):
            st.session_state.active_pdf = fname
    if st.session_state.active_pdf:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### Content of `{st.session_state.active_pdf}`")
        st.sidebar.text_area(
            "📜 Text Preview",
            st.session_state.pdf_texts[st.session_state.active_pdf]["text"],
            height=300,
            key=f"text_preview_{st.session_state.active_pdf}"
        )

st.title("🧠 Chat with LLaMA + PDFs (LangChain Memory)")

for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

user_query = st.chat_input("Ask a question...")

if user_query and user_query.strip():
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append(("user", user_query))
    answer = answer_query(user_query)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append(("assistant", answer))
