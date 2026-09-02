# v6_app.py
# Visible "Thinking Mode" Toggle
# Hide on Cloud Mode
# Passed Configuration to Backend
# Real-time Thinking Status Box


import uuid
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from v6_backend import (
    chatbot, 
    ingest_pdf, 
    retrieve_all_threads, 
    thread_document_metadata,
    set_thread_settings,
    get_thread_settings
)

os.environ["LANGCHAIN_PROJECT"] = "Chatbot Project"


# ==============================================================================
# 1. TEXT EXTRACTION HELPER
# ==============================================================================
def extract_text(content):
    """Safely extracts plain string text from AIMessage content."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                val = item.get("text") or item.get("text_delta") or item.get("content") or ""
                text_parts.append(val)
        return "".join(text_parts)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    return str(content)


# ==============================================================================
# 2. SESSION UTILITY FUNCTIONS
# ==============================================================================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    set_thread_settings(thread_id, is_private=False, thinking_mode=False)
    add_threads(thread_id)
    st.session_state["message_history"] = []

def add_threads(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


# ==============================================================================
# 3. SESSION STATE INITIALIZATION
# ==============================================================================
if 'message_history' not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    set_thread_settings(st.session_state["thread_id"], is_private=False, thinking_mode=False)

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_threads(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
is_private, thinking_mode = get_thread_settings(thread_key)
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ==============================================================================
# 4. SIDEBAR UI
# ==============================================================================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

active_thread_pdf = thread_document_metadata(thread_key)

if active_thread_pdf:
    st.sidebar.success(
        f"📄 **Indexed PDF:** `{active_thread_pdf.get('filename')}`\n\n"
        f"({active_thread_pdf.get('chunks')} chunks from {active_thread_pdf.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed for this chat thread.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat thread", 
    type=["pdf"], 
    key=f"pdf_uploader_{thread_key}"
)
if uploaded_pdf:
    if active_thread_pdf and active_thread_pdf.get("filename") == uploaded_pdf.name:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed for this thread.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                    is_private=is_private
                )
                status_box.update(label="✅ PDF indexed for this thread", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                st.sidebar.error(f"Error: {str(e)}")

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        t_priv, _ = get_thread_settings(tid)
        priv_label = "🔒" if t_priv else "☁️"
        if st.sidebar.button(f"{priv_label} {tid[:8]}...", key=f"side-thread-{tid}"):
            selected_thread = tid


# ==============================================================================
# 5. MAIN CHAT UI
# ==============================================================================
st.title("Multi Utility Chatbot")

# Detect whether the first message is already entered or in the active submission turn
chat_has_started = (
    len(st.session_state["message_history"]) > 0 
    or bool(st.session_state.get("chat_input_box"))
)

# Mode selection only shown BEFORE the user hits enter on message 1
if not chat_has_started:
    st.info("💡 **Mode Selection for New Chat:**")
    selected_mode = st.radio(
        "Choose pipeline mode for this conversation:",
        options=[
            "☁️ Standard Cloud Mode (Gemini 3.5 Flash & Gemini Embeddings)",
            "🔒 Private Mode (Local Qwen 3.5 & Local Nomic Embeddings - 100% Offline)"
        ],
        index=1 if is_private else 0,
        key=f"radio_mode_{thread_key}"
    )
    chosen_private = "🔒 Private Mode" in selected_mode

    chosen_thinking = False
    if chosen_private:
        chosen_thinking = st.toggle(
            "Thinking Mode",
            value=thinking_mode,
            key=f"toggle_think_{thread_key}",
            help="Enables deep step-by-step reasoning for local Qwen"
        )

    set_thread_settings(thread_key, chosen_private, chosen_thinking)
    is_private, thinking_mode = chosen_private, chosen_thinking
else:
    # Permanently locked read-only indicators once first message is sent
    mode_text = "🔒 Private Mode (Local Qwen 3.5)" if is_private else "☁️ Standard Cloud Mode (Gemini)"
    if is_private:
        think_text = "ON" if thinking_mode else "OFF"
        st.caption(f"Session Mode: **{mode_text}** | Thinking Mode: **{think_text}**")
    else:
        st.caption(f"Session Mode: **{mode_text}**")

# Display conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here..", key="chat_input_box")

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

if user_input:
    st.session_state["message_history"].append({"role": "human", "content": user_input})
    with st.chat_message('human'):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        tool_status = {"box": None}
        think_holder = {"box": None}

        # Show status bar only when local Qwen runs with thinking mode ON
        if is_private and thinking_mode:
            think_holder["box"] = st.status("Thinking…", expanded=False)

        def ai_only_stream():
            in_think_block = False
            raw_buffer = ""

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # 1. Handle Tool Execution
                if isinstance(message_chunk, ToolMessage):
                    if think_holder["box"] is not None:
                        think_holder["box"].update(label="Thinking finished", state="complete", expanded=False)
                        think_holder["box"] = None

                    tool_name = getattr(message_chunk, "name", "tool")
                    if tool_status["box"] is None:
                        tool_status["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        tool_status["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                # 2. Filter thinking tokens and stream clean response
                if isinstance(message_chunk, AIMessage):
                    chunk_text = extract_text(message_chunk.content)
                    if not chunk_text:
                        continue

                    raw_buffer += chunk_text

                    while raw_buffer:
                        if in_think_block:
                            if "</think>" in raw_buffer:
                                _, raw_buffer = raw_buffer.split("</think>", 1)
                                in_think_block = False
                                if think_holder["box"] is not None:
                                    think_holder["box"].update(label="Thinking finished", state="complete", expanded=False)
                                    think_holder["box"] = None
                            else:
                                raw_buffer = ""
                        else:
                            if "<think>" in raw_buffer:
                                text_before, raw_buffer = raw_buffer.split("<think>", 1)
                                in_think_block = True
                                if text_before:
                                    yield text_before
                            else:
                                if think_holder["box"] is not None:
                                    think_holder["box"].update(label="Thinking finished", state="complete", expanded=False)
                                    think_holder["box"] = None

                                yield raw_buffer
                                raw_buffer = ""

            if think_holder["box"] is not None:
                think_holder["box"].update(label="Thinking finished", state="complete", expanded=False)

        ai_message = st.write_stream(ai_only_stream())

        if tool_status["box"] is not None:
            tool_status["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        st.session_state["message_history"].append({
            "role": "ai", 
            "content": ai_message
        })

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (SystemMessage, ToolMessage)):
            continue

        if isinstance(msg, HumanMessage):
            text = extract_text(msg.content)
            if text:
                temp_messages.append({"role": "human", "content": text})

        elif isinstance(msg, AIMessage):
            text = extract_text(msg.content)
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>", 1)[-1].strip()
            if text:
                temp_messages.append({"role": "ai", "content": text})

    st.session_state["message_history"] = temp_messages
    st.rerun()