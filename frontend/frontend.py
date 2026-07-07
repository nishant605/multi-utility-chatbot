import os
import sys
from pathlib import Path
import bcrypt

# Get the project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend.backend import (app, pdf_process, retrieve_all_threads, thread_document_metadata, UPLOAD_DIR,
                              delete_thread, get_thread_title, set_thread_title, delete_thread_title,
                              generate_thread_title, register_user, login_user)
from streamlit_local_storage import LocalStorage

# ----------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

locals = LocalStorage()

# ----------------------------------------------------------------------------
# Custom styling — clean white theme
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ---------- Global ---------- */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #ffffff !important;
        color: #1a1a2e;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #f7f8fa !important;
        border-right: 1px solid #eaeaea;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a2e;
    }

    /* ---------- Titles ---------- */
    h1, h2, h3 {
        color: #1a1a2e !important;
        font-weight: 700;
    }
    h1 {
        letter-spacing: -0.5px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background-color: #ffffff;
        color: #4f46e5;
        border: 1.5px solid #4f46e5;
        border-radius: 10px;
        padding: 0.4rem 1rem;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #4f46e5;
        color: #ffffff;
        border-color: #4f46e5;
    }

    /* Primary "New Chat" style button (first sidebar button) */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border: 1.5px solid #e0e0e0;
        border-radius: 14px;
        background-color: #fafafa;
    }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        background-color: #f7f8fa;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid #efefef;
    }

    /* ---------- Alerts (success / warning / info) ---------- */
    div[data-baseweb="notification"], .stAlert {
        border-radius: 10px !important;
    }

    /* ---------- Text inputs ---------- */
    input, textarea {
        border-radius: 10px !important;
        background-color: #fafafa !important;
        color: #1a1a2e !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: #eaeaea;
    }

    /* ---------- Status widget ---------- */
    [data-testid="stStatusWidget"] {
        border-radius: 10px;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #d0d0d8;
        border-radius: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)
    st.session_state.message_history = []


def load_conversation_history(thread_id):
    state = app.get_state(config={"configurable": {"thread_id": thread_id, "user_id": st.session_state.user_id}})
    return state.values.get("messages", [])


# ----------------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = {}

if "ls_loaded" not in st.session_state:
    st.session_state.ls_loaded = False

if "user_id" not in st.session_state:
    stored_user_id = locals.getItem("user_id")
    if stored_user_id is None and not st.session_state.ls_loaded:
        st.session_state.ls_loaded = True
        st.rerun()
    st.session_state.user_id = stored_user_id
    st.session_state.ls_loaded = True

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrieve_all_threads(st.session_state.user_id)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = st.session_state.user_id is not None

add_thread(st.session_state.thread_id)

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_files"].setdefault(thread_key, {})
threads = st.session_state["chat_thread"][::-1]

# ----------------------------------------------------------------------------
# Login / Register screen
# ----------------------------------------------------------------------------
if not st.session_state.logged_in:
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome back")
        st.caption("Sign in or create an account to continue")

        option = st.radio("Choose an option", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if option == "Register":
            if st.button("Create account", use_container_width=True):
                if register_user(username, password):
                    st.success("✅ Registration successful! Please log in.")
                else:
                    st.error("⚠️ Username already exists. Please choose a different username.")
        else:
            if st.button("Login", use_container_width=True):
                user_id = login_user(username, password)
                if user_id:
                    st.session_state.user_id = username
                    st.session_state.logged_in = True
                    locals.setItem("user_id", username)
                    st.session_state.chat_thread = retrieve_all_threads(username)
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("⚠️ Invalid username or password.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 💬 LangGraph PDF Chatbot")
st.sidebar.caption(f"Thread: `{thread_key[:8]}…`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(f"📄 Using `{latest_doc.get('filename')}`")
    st.sidebar.info(f"{latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages")
else:
    st.sidebar.warning("📭 No PDF uploaded yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.warning(f"File `{uploaded_pdf.name}` already uploaded in this thread.")
    else:
        with st.sidebar.status("Processing PDF...") as statusbox:
            save_path = UPLOAD_DIR / uploaded_pdf.name
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())
            summary = pdf_process(save_path, thread_key, uploaded_pdf.name)
            thread_docs[uploaded_pdf.name] = summary
            statusbox.success(f"PDF `{uploaded_pdf.name}` processed successfully.")

st.sidebar.divider()
st.sidebar.subheader("🗂️ Past conversations")

selected_thread_id = None

if not threads:
    st.sidebar.caption("No past conversations found.")
else:
    for thread_id in threads:
        display_name = get_thread_title(thread_id) or f"Thread {thread_id[:8]}"
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if col1.button(display_name, key=f"thread_{thread_id}", use_container_width=True):
                selected_thread_id = thread_id
        with col2:
            if col2.button("🗑️", key=f"delete_{thread_id}"):
                delete_thread(thread_id)
                st.session_state.chat_thread.remove(thread_id)
                st.session_state.ingested_files.pop(str(thread_id), None)

                if thread_id == st.session_state.thread_id:
                    reset_chat()

                st.rerun()

st.sidebar.divider()

# ----------------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------------
st.title("💬 Multi Utility Chatbot")
st.caption("Ask about your document, or use any of the connected tools")

for message in st.session_state.message_history:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    if get_thread_title(thread_key) is None:
        set_thread_title(thread_key, generate_thread_title(user_input))

    CONFIG = {
        "configurable": {"thread_id": thread_key, "user_id": st.session_state.user_id},
        "metadata": {"thread_id": thread_key, "user_id": st.session_state.user_id},
        "run_name": "chat_turn",
        "tags": ["streamlit", "ltm", "rag", "multi-utility"],
    }

    with st.chat_message("assistant", avatar="🤖"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_response = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state.message_history.append({"role": "assistant", "content": ai_response})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.sidebar.subheader("📊 Document Metadata")
        st.sidebar.markdown(f"**{doc_meta.get('filename')}**")
        st.sidebar.markdown(f"- Pages: {doc_meta.get('documents')}")
        st.sidebar.markdown(f"- Chunks: {doc_meta.get('chunks')}")

st.divider()

if selected_thread_id:
    st.session_state.thread_id = selected_thread_id
    message = load_conversation_history(selected_thread_id)

    temp_message_history = []
    for msg in message:
        if isinstance(msg, HumanMessage):
            temp_message_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            temp_message_history.append({"role": "assistant", "content": msg.content})
    st.session_state.message_history = temp_message_history
    st.session_state["ingested_files"].setdefault(str(selected_thread_id), {})
    st.rerun()

if st.sidebar.button("🚪 Logout", use_container_width=True):
    locals.deleteItem("user_id")
    st.session_state.clear()
    st.rerun()