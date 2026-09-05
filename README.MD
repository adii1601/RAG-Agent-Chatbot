# 🤖 Agentic RAG Chatbot with Tool Calling

An Agentic Retrieval-Augmented Generation (RAG) assistant built with LangGraph, LangChain, FAISS, and Streamlit. The agent intelligently selects between dynamic web search and semantic document retrieval, supporting seamless toggling between local LLMs (Ollama) and cloud models (Google Gemini).

### 🚀 Live Demo

👉 **[Try the Live App](https://rag-agent-chatbot-adii.streamlit.app/)**

---

## ✨ Features
* **Dual Model Support**: Toggle between local privacy-first models via Ollama and cloud-hosted Google Gemini.
* **Autonomous Tool Routing**: Employs LangGraph agentic workflows to decide when to retrieve from documents or search external web sources via DuckDuckGo.
* **Semantic Document Ingestion**: Ingests and chunks PDF documents, generating vector embeddings stored locally in FAISS.
* **Interactive UI**: Built with Streamlit for clean conversational interaction and real-time tool execution tracking.
* **Markdown-Aware Table Extraction**: Uses PyMuPDF4LLM to convert PDF tables into structured Markdown grids (`| col | col |`) rather than flattening them into unstructured, misaligned text.
* **Table-Preserving Chunking**: Employs hierarchical separators (`\n## `, `\n### `, `\n\n`) to prevent cutting tables in half across chunk boundaries, ensuring complete headers and data stay intact.

---

## 🛠️ Tech Stack
* **Language & Frameworks**: Python, Streamlit
* **Agent Orchestration**: LangGraph, LangChain
* **Models**: Google Gemini 3.5 Flash (Cloud), Ollama Qwen 3.5 (Offline Private Mode)
* **Embeddings & Vector Store**: Gemini Embeddings / Nomic Embed, FAISS
* **Document Processing & OCR**: PyMuPDF4LLM (Markdown table parsing & structured layout extraction)
* **Text Chunking**: RecursiveCharacterTextSplitter (structure- and table-preserving separators)
* **Storage & Checkpointing**: SQLite, LangGraph SqliteSaver (thread-level conversation memory)
* **Agent Tools**: Document RAG (`rag_tool`), DuckDuckGo Web Search, Alpha Vantage Stock API, Arithmetic Calculator

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/adii1601/RAG-Agent-Chatbot.git
cd RAG-Agent-Chatbot
```
### 2. Set up virtual environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Configure environment variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
### 4. Launch the application
```bash
streamlit run app.py
```