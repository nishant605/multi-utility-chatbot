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
from backend.backend import (app,pdf_process,retrieve_all_threads,thread_document_metadata,UPLOAD_DIR,delete_thread,get_thread_title,set_thread_title,delete_thread_title,generate_thread_title
                             ,register_user,login_user)
from streamlit_local_storage import LocalStorage

locals = LocalStorage()

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
    state = app.get_state(config={"configurable": {"thread_id": thread_id,"user_id": st.session_state.user_id}})
    return state.values.get("messages", [])

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = {}

if "ls_loaded" not in st.session_state:
    st.session_state.ls_loaded = False

if "user_id" not in st.session_state:
    stored_user_id = locals.getItem("user_id")  # You can replace this with a more sophisticated user management system
    if stored_user_id is None and not st.session_state.ls_loaded:
        st.session_state.ls_loaded = True
        st.rerun()
    st.session_state.user_id = stored_user_id
    st.session_state.ls_loaded = True

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrieve_all_threads(st.session_state.user_id)  # Retrieve all threads for the user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = st.session_state.user_id is not None

add_thread(st.session_state.thread_id)

thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state["ingested_files"].setdefault(thread_key, {})
threads = st.session_state["chat_thread"][::-1]  # Reverse the order of threads for display

if not st.session_state.logged_in:
    st.title("🔐 Login")
    option = st.radio("Choose an option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registration successful! Please log in.")
            else:
                st.error("Username already exists. Please choose a different username.")
    else:
        if st.button("Login"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state.user_id = username 
                st.session_state.logged_in = True
                locals.setItem("user_id", username)  # Store the user_id in local storage
                st.session_state.chat_thread = retrieve_all_threads(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()  # Stop further execution until the user logs in

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat",use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(f"Using `{latest_doc.get('filename')}`")
    st.sidebar.info(f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)")
else:
    st.sidebar.warning("No PDF uploaded yet.")
    
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

st.sidebar.subheader("Past conversations")

selected_thread_id = None

if not threads:
    st.sidebar.write("No past conversations found.")
else:
    for thread_id in threads:
        display_name = get_thread_title(thread_id) or f"Thread {thread_id[:8]}"
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if col1.button(display_name, key=f"thread_{thread_id}"):
                selected_thread_id = thread_id
        with col2:
            if col2.button("❌", key=f"delete_{thread_id}"):
                delete_thread(thread_id)
                st.session_state.chat_thread.remove(thread_id)
                st.session_state.ingested_files.pop(str(thread_id), None)

                if thread_id == st.session_state.thread_id:
                    reset_chat()

                st.rerun()

st.title("Multi Utility Chatbot")

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.text(message["content"])
    
user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    if get_thread_title(thread_key) is None:
        set_thread_title(thread_key, generate_thread_title(user_input))

    CONFIG = {
        "configurable": {"thread_id":thread_key,"user_id": st.session_state.user_id},
        "metadata": {"thread_id": thread_key,"user_id": st.session_state.user_id},
        "run_name" : "chat_turn",
        "tags" : ["streamlit", "ltm", "rag", "multi-utility"],
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk,_ in app.stream(
                {"messages":[HumanMessage(content=user_input)]},
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
        st.sidebar.subheader("Document Metadata")
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

if st.sidebar.button("logout"):
    locals.deleteItem("user_id")  # Remove the user_id from local storage
    st.session_state.clear()
    st.rerun()