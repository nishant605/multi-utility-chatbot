#imports

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage,HumanMessage
from langchain_core.tools import tool
from typing import Annotated, Any, Dict, Optional, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import date
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import yfinance as yf
from langchain_core.runnables import RunnableConfig
import shutil
from pathlib import Path
from langgraph.store.sqlite import SqliteStore
import re
import bcrypt

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DB_DIR = BASE_DIR / "database"
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
DB_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Database paths
CHAT_MEMORY_DB = DB_DIR / "chat_memory.db"
EXPENSE_DB = DB_DIR / "expenses.db"

MEMORY_DB = DB_DIR / "memories.db"
USER_DB = DB_DIR / "users.db"

#models

llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))
embedding = CohereEmbeddings(model="embed-v4.0",cohere_api_key=os.getenv("COHERE_API_KEY"))

today_str = date.today().isoformat()  # Get today's date in YYYY-MM-DD format

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching stock price: {response.status_code}")
    return response.json()

@tool
def get_indian_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'TATASTEEL.NS', 'RELIANCE.NS') 
    using Yahoo Finance.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "symbol": info.get("symbol"),
            "price": info.get("regularMarketPrice"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }
    except Exception as e:
        raise Exception(f"Error fetching Indian stock price: {e}")

conn1 = sqlite3.connect(EXPENSE_DB)  # Connect to the expenses database 
cursor = conn1.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    date TEXT NOT NULL
)
""")

conn1.commit()
conn1.close()

@tool
def add_expense(config: RunnableConfig, amount: float, category: str, description: Optional[str], expense_date: str) -> dict:
    """
    Add a new expense to the database.
    """
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        return {"status": "error", "message": "No user_id found in session."}
    
    if expense_date is None:
        expense_date = date.today().isoformat()

    if amount <= 0:
        return {"status": "error", "message": "Amount must be greater than zero."}
    
    conn = sqlite3.connect(EXPENSE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO expenses (user_id, amount, category, description, date)
    VALUES (?, ?, ?, ?, ?)
    """, (user_id, amount, category, description, expense_date))
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Expense added successfully."}

@tool
def show_expenses(config: RunnableConfig) -> dict:
    """
    Show all expenses for a given user.
    """
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        return {"status": "error", "message": "No user_id found in session."}

    conn = sqlite3.connect(EXPENSE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT * FROM expenses WHERE user_id = ?
    """, (user_id,))
    
    expenses = cursor.fetchall()
    conn.close()
    
    return {"status": "success", "expenses": expenses}

@tool
def monthly_expense_summary(config: RunnableConfig, month: str) -> dict:
    """
    Show a summary of expenses for a given user and month (format: 'YYYY-MM').
    """
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        return {"status": "error", "message": "No user_id found in session."}

    conn = sqlite3.connect(EXPENSE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT category, SUM(amount) as total_amount
    FROM expenses
    WHERE user_id = ? AND strftime('%Y-%m', date) = ?
    GROUP BY category
    """, (user_id, month))
    
    summary = cursor.fetchall()
    conn.close()
    
    return {"status": "success", "summary": summary}

@tool
def delete_expense(expense_id: int) -> dict:
    """
    Delete an expense by its ID.
    """
    conn = sqlite3.connect(EXPENSE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
    DELETE FROM expenses WHERE id = ?
    """, (expense_id,))
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Expense deleted successfully."}

@tool
def update_expense(expense_id: int, amount: Optional[float] = None, category: Optional[str] = None, description: Optional[str] = None, expense_date: Optional[str] = None) -> dict:
    """
    Update an existing expense by its ID. Only provided fields will be updated.
    """
    conn = sqlite3.connect(EXPENSE_DB)
    cursor = conn.cursor()
    
    # Build the update query dynamically based on provided fields
    fields_to_update = []
    values = []
    
    if amount is not None:
        fields_to_update.append("amount = ?")
        values.append(amount)
    if category is not None:
        fields_to_update.append("category = ?")
        values.append(category)
    if description is not None:
        fields_to_update.append("description = ?")
        values.append(description)
    if expense_date is not None:
        fields_to_update.append("date = ?")
        values.append(expense_date)
    
    if not fields_to_update:
        return {"status": "error", "message": "No fields to update."}
    
    values.append(expense_id)
    update_query = f"UPDATE expenses SET {', '.join(fields_to_update)} WHERE id = ?"
    
    cursor.execute(update_query, tuple(values))
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Expense updated successfully."}

_THREAD_RETRIEVER : Dict[str, Any] = {}
_THREAD_METADATA : Dict[str, dict] = {}

def _get_retriever(thread_id : Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVER:
        return _THREAD_RETRIEVER[thread_id]
    return None

def pdf_process(file_path: str, thread_id: str, filename: Optional[str] = None) -> dict:
        """
        Build a FAI SS retriever for the uploaded PDF and store it for the thread.

        Returns a summary dict that can be surfaced in the UI.
        """
        #computer read from bytes
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
            chunks = splitter.split_documents(docs)

            # Create a FAISS vector store from the chunks
            vector_store = FAISS.from_documents(chunks, embedding)

            vector_path = VECTORSTORE_DIR / thread_id

            vector_store.save_local(str(vector_path))

            retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
            )

            _THREAD_RETRIEVER[thread_id] = retriever
            _THREAD_METADATA[thread_id] = {"filename": filename or os.path.basename(file_path),
                                        "documents": len(docs),
                                        "chunks": len(chunks)}

            return {
                "filename": filename or os.path.basename(file_path),
                "documents": len(docs),
                "chunks": len(chunks),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()   # prints full stack trace to your terminal/logs
            return {"error": str(e)}
    
@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for the current chat thread.
    """

    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No retriever found for the given thread_id. Please upload a PDF first."}
    
    results = retriever.invoke(query)
    context = [doc.page_content for doc in results]
    metadata = [doc.metadata for doc in results]

    return { 
        "query" : query,
        "context" : context,
        "metadata" : metadata,
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename")
    }

@tool
def remember(config: RunnableConfig,memory:str):
    """
    Save long-term information about the user.
    """

    user_id = config["configurable"]["user_id"]
    conn = sqlite3.connect(MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
                   INSERT INTO memories (user_id, memory) VALUES (?, ?)
                   """, (user_id, memory))
    conn.commit()
    conn.close()
    return "memory saved"


tools = [search_tool,get_stock_price, add_expense, show_expenses, monthly_expense_summary, delete_expense, update_expense,rag_tool,get_indian_stock_price,remember]
llm_with_tools = llm.bind_tools(tools)

conn = sqlite3.connect(CHAT_MEMORY_DB,check_same_thread=False) #false to allow access from multiple threads
memory = SqliteSaver(conn=conn)

conn = sqlite3.connect(MEMORY_DB)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    memory TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

conn = sqlite3.connect(USER_DB)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
""")

conn.commit()
conn.close()

def register_user(username: str, password: str):
    if not username or not password:
        return False  # Invalid input
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password.decode('utf-8')))
        conn.commit()   
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    
def login_user(username: str, password: str):
    if not username or not password:
        return False  # Invalid input
    conn = sqlite3.connect(USER_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE username = ? ", (username,))
    result = cursor.fetchone()
    conn.close()

    if result is None:
        return False
    stored_hash = result[0].encode("utf-8")
    if bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash
    ):
        return username

    return False

def delete_thread(thread_id: str) -> dict:
    """
    Delete a thread and its associated data from memory and vector store.
    """
    # Remove from in-memory retriever and metadata
    try:
        del_conn = sqlite3.connect(CHAT_MEMORY_DB)
        cursor = del_conn.cursor()
        for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs","thread_owners"):
            try:
                cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass  # table may not exist depending on langgraph version
        del_conn.commit()
        del_conn.close()


        vector_path = VECTORSTORE_DIR / thread_id
        if vector_path.exists():
            shutil.rmtree(vector_path)

        _THREAD_RETRIEVER.pop(thread_id, None)
        _THREAD_METADATA.pop(thread_id, None)
        delete_thread_title(thread_id)
        return {"status": "success", "message": f"Thread {thread_id} deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def _init_titles_table():
    """
    Initialize the titles table in the database if it doesn't exist.
    """
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

_init_titles_table()

def get_thread_title(thread_id: str) -> Optional[str]:
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM thread_titles WHERE thread_id = ?", (thread_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_thread_title(thread_id: str, title: str) -> None:
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO thread_titles (thread_id, title) VALUES (?, ?)
    ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title
    """, (thread_id, title))
    conn.commit()
    conn.close()

def delete_thread_title(thread_id: str) -> None:
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()

def generate_thread_title(first_message: str) -> str:
    """
    Use the LLM to generate a title for the thread based on its content.
    """
    try:
        response = llm.invoke([SystemMessage(content=(
                """Summarize the following user message into a short chat title.
                Rules: 5 words or fewer, no punctuation, no quotes, no markdown, 
                title case. Just output the title, nothing else.","You have access to long-term memory.When the user tells you stable personal information
(name, favourite things, hometown, preferences,
birth date, goals, occupation, etc.)
call remember.

Before answering questions about the user,
call recall"""
)),
            HumanMessage(content=first_message),])
        title = response.content
        title = re.sub(r"<think>.*?</think>", "", title, flags=re.DOTALL)
        title = title.strip().strip('"').strip("'") 
        return title[:40]  # Limit title to 40 characters
    except Exception as e:
        return (first_message[:40] + "...") if len(first_message) > 40 else first_message
    
def get_user_memories(user_id: str) -> str:
    """
    Retrieve all long-term memories for the user.
    """

    conn = sqlite3.connect(MEMORY_DB)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT memory
        FROM memories
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20
    """, (user_id,))

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return "No long-term memories."

    return "\n".join(f"- {row[0]}" for row in rows)

def _init_thread_owner_table():
    """
    Initialize the thread owner table in the database if it doesn't exist.
    """
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_owners (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

_init_thread_owner_table()

def set_thread_owner(thread_id: str, user_id: str) -> None:
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO thread_owners (thread_id, user_id) VALUES (?, ?)
    ON CONFLICT(thread_id) DO NOTHING
    """, (thread_id, user_id))
    conn.commit()
    conn.close()

class ChatState(TypedDict):
   messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState, config=None) -> ChatState:
    """Main chatbot node."""

    user_id = config.get("configurable", {}).get("user_id")
    thread_id = config.get("configurable", {}).get("thread_id")
    if user_id and thread_id:
        set_thread_owner(thread_id, user_id)
        

    memories = get_user_memories(user_id)

    system_message = SystemMessage(
        content=f"""
You are a helpful AI assistant.

Today's date is {today_str}.

==============================
LONG TERM MEMORY
==============================

{memories}

==============================
RULES
==============================

1. Use the long-term memory whenever it is relevant.

2. If the user asks about themselves,
answer using the stored memories.

3. If the user tells you NEW stable information
such as:

- Name
- Birthday
- Age
- Hometown
- City
- Occupation
- College
- Favourite food
- Favourite color
- Preferences
- Goals
- Family members
- Permanent facts   

call the remember tool.

4. For questions about uploaded PDFs,
call rag_tool.

5. For expense tracking,
use add_expense,
show_expenses,
monthly_expense_summary,
update_expense,
delete_expense.

Never ask for a user_id.
"""
    )

    messages = [system_message] + state["messages"][-10:]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {"messages": [response]}

tool_node = ToolNode(tools) #"Here is a list of Python functions (tools). If the LLM asks to use one of them, execute it."  

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node", tools_condition)  # Add conditional edges based on tool usage
graph.add_edge("tools", "chat_node")  # Loop back to chat_node after tool execution
app = graph.compile(checkpointer=memory)

def retrieve_all_threads(user_id):
    if not user_id:
        return []
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM thread_owners WHERE user_id = ?", (user_id,))
    threads = [row[0] for row in cursor.fetchall()]
    conn.close()
    return threads

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})