# PDF-CHATBOT
# 🧠 PDF + General Chatbot with LLaMA (LangChain + Streamlit)

This project lets you **chat with uploaded PDFs** or ask **general questions** using a **local LLaMA model**, powered by **LangChain**, **Ollama**, and **Streamlit** — all running seamlessly inside Docker.

---

## 🚀 Features

- 🗂️ Chat with any uploaded PDF (context-aware Q&A)
- 💬 Ask general ChatGPT-style questions
- 🧠 LangChain memory keeps chat history
- 🖼️ Automatic OCR for scanned PDFs
- 🌐 Clean Streamlit UI
- 🐳 Fully containerized — no Python or Ollama setup required

---

## 📦 Requirements

- ✅ [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

> 💡 No need to install Python or Ollama manually. Everything runs inside containers.

---

## 🛠️ How to Run (1-Line Setup)

### 🧩 Step 1: Clone & Launch

```bash
git clone https://github.com/Dhanvanyaa/PDF-CHATBOT.git
cd PDF-CHATBOT
docker compose up --build
