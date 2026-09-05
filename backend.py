from __future__ import annotations

# ==============================================================================
# 1. IMPORTS & ENVIRONMENT SETUP
# ==============================================================================
import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

# Document Loading (Markdown + OCR) & Vectorstores
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embedding Models & LLMs
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# LangGraph Components & Checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"


# ==============================================================================
# 2. MODEL & EMBEDDING INITIALIZATION
# ==============================================================================

# --- CLOUD PIPELINE (Standard Mode) ---
cloud_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
cloud_embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# --- LOCAL PIPELINE (Private Mode) ---
local_llm = ChatOllama(model="hf.co/unsloth/Qwen3.5-9B-GGUF:Q3_K_S")
local_embedding = OllamaEmbeddings(model="nomic-embed-text")


# ==============================================================================
# 3. SQLITE SETTINGS DATABASE
# ==============================================================================
DB_PATH = "chatbot.db"

def init_settings_db():
    """Initializes table for persisting thread privacy & thinking settings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_settings (
            thread_id TEXT PRIMARY KEY,
            is_private INTEGER NOT NULL,
            thinking_mode INTEGER DEFAULT 0
        )
    """)
    cursor.execute("PRAGMA table_info(thread_settings)")
    cols = [c[1] for c in cursor.fetchall()]
    if "thinking_mode" not in cols:
        cursor.execute("ALTER TABLE thread_settings ADD COLUMN thinking_mode INTEGER DEFAULT 0")
    conn.commit()
    conn.close()

init_settings_db()

def set_thread_settings(thread_id: str, is_private: bool, thinking_mode: bool = False):
    """Saves thread settings in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO thread_settings (thread_id, is_private, thinking_mode)
        VALUES (?, ?, ?)
    """, (str(thread_id), 1 if is_private else 0, 1 if thinking_mode else 0))
    conn.commit()
    conn.close()

def get_thread_settings(thread_id: str) -> tuple[bool, bool]:
    """Retrieves (is_private, thinking_mode) for a thread."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT is_private, thinking_mode FROM thread_settings WHERE thread_id = ?", (str(thread_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return bool(row[0]), bool(row[1])
    return False, False


# ==============================================================================
# 4. STRICT THREAD-LEVEL PDF RETRIEVER MANAGEMENT
# ==============================================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


import fitz  # PyMuPDF engine
import pymupdf4llm
from langchain_core.documents import Document

_OCR_ENGINE = None

def _get_ocr_engine():
    """Lazy-loads RapidOCR only when a scanned document is detected."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception:
            try:
                from rapidocr import RapidOCR
                _OCR_ENGINE = RapidOCR()
            except Exception:
                _OCR_ENGINE = None
    return _OCR_ENGINE

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None, is_private: bool = False) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        resolved_filename = filename or os.path.basename(temp_path)
        docs = []
        ocr_engine = _get_ocr_engine()

        # Open PDF with PyMuPDF to inspect each page
        with fitz.open(temp_path) as pdf_doc:
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc[page_idx]
                page_text = ""

                # 1. First attempt: Structured markdown extraction
                try:
                    page_text = pymupdf4llm.to_markdown(pdf_doc, pages=[page_idx], table_strategy="lines")
                except Exception:
                    page_text = ""

                # 2. Fallback: If page is scanned (less than 40 chars extracted), run RapidOCR
                if (not page_text or len(page_text.strip()) < 40) and ocr_engine is not None:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        ocr_result, _ = ocr_engine(img_bytes)
                        if ocr_result:
                            ocr_lines = [line[1] for line in ocr_result if len(line) > 1]
                            page_text = "\n".join(ocr_lines)
                    except Exception:
                        pass

                if page_text and page_text.strip():
                    docs.append(
                        Document(
                            page_content=page_text.strip(),
                            metadata={
                                "source": resolved_filename,
                                "filename": resolved_filename,
                                "page": page_idx + 1
                            }
                        )
                    )

        if not docs:
            raise ValueError("The PDF could not be read or contains no extractable text/scans.")

        # Structure-aware chunking preserving markdown table integrity
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        chosen_embedding = local_embedding if is_private else cloud_embedding
        vector_store = FAISS.from_documents(chunks, chosen_embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": resolved_filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "is_private": is_private,
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None, is_private: bool = False) -> dict:
#     if not file_bytes:
#         raise ValueError("No bytes received for ingestion.")

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(file_bytes)
#         temp_path = temp_file.name

#     try:
#         # Load documents as markdown and run RapidOCR on scanned pages/images
#         loader = PyMuPDF4LLMLoader(
#             temp_path,
#             mode="page",
#             table_strategy="lines",
#             extract_images=True,
#             images_parser=RapidOCRBlobParser()
#         )
#         docs = loader.load()

#         if not docs:
#             raise ValueError("The PDF could not be read or contains no text/image pages.")

#         # Retain original uploaded filename in metadata
#         resolved_filename = filename or os.path.basename(temp_path)
#         for doc in docs:
#             doc.metadata["source"] = resolved_filename
#             doc.metadata["filename"] = resolved_filename

#         # Structure-aware chunking preserving markdown table integrity
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000,
#             chunk_overlap=200,
#             separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
#         )
#         chunks = splitter.split_documents(docs)

#         if not chunks:
#             raise ValueError("No extractable text found in this PDF.")

#         chosen_embedding = local_embedding if is_private else cloud_embedding
#         vector_store = FAISS.from_documents(chunks, chosen_embedding)
#         retriever = vector_store.as_retriever(
#             search_type="similarity", search_kwargs={"k": 5}
#         )

#         _THREAD_RETRIEVERS[str(thread_id)] = retriever
#         _THREAD_METADATA[str(thread_id)] = {
#             "filename": resolved_filename,
#             "documents": len(docs),
#             "chunks": len(chunks),
#             "is_private": is_private,
#         }

#         return _THREAD_METADATA[str(thread_id)]
#     finally:
#         try:
#             os.remove(temp_path)
#         except OSError:
#             pass


# ==============================================================================
# 5. TOOLS DEFINITION
# ==============================================================================

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread. 
    Context is returned formatted in Markdown, preserving tables and section headers."""
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No PDF document uploaded for this specific chat thread. Please upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
cloud_llm_with_tools = cloud_llm.bind_tools(tools)
local_llm_with_tools = local_llm.bind_tools(tools)


# ==============================================================================
# 6. LANGGRAPH STATE & DYNAMIC CHAT NODE
# ==============================================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    is_private, thinking_mode = get_thread_settings(thread_id) if thread_id else (False, False)

    mode_label = "PRIVATE" if is_private else "STANDARD"

    system_prompt = (
        f"You are a helpful AI assistant running in {mode_label} mode.\n"
        "For questions regarding the uploaded document or PDF, use the `rag_tool`.\n"
        "Document context returned by `rag_tool` is formatted in Markdown. "
        "Tables appear as `| column | column |` markdown rows; analyze them row by row when answering."
    )

    if is_private:
        if thinking_mode:
            system_prompt += "\n/think"
        else:
            system_prompt += "\n/no_think"

    system_message = SystemMessage(content=system_prompt)
    recent_messages = state["messages"][-8:] if len(state["messages"]) > 8 else state["messages"]
    messages = [system_message, *recent_messages]

    active_llm = local_llm_with_tools if is_private else cloud_llm_with_tools
    response = active_llm.invoke(messages, config=config)

    if is_private and isinstance(response.content, str):
        lowered = response.content.lower()
        if "don't have a specific function call" in lowered or "no function call" in lowered:
            response.content = "Hello! How can I help you today?"

    return {"messages": [response]}

tool_node = ToolNode(tools)


# ==============================================================================
# 7. CHECKPOINTER & GRAPH COMPILATION
# ==============================================================================

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ==============================================================================
# 8. EXPORTED HELPERS
# ==============================================================================

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})