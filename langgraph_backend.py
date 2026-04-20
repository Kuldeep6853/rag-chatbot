from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    streaming=True,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# -------------------
# 2. PDF retriever store (Global)
# -------------------
GLOBAL_RETRIEVER = None
GLOBAL_METADATA = {}


def _init_global_retriever():
    global GLOBAL_RETRIEVER, GLOBAL_METADATA
    if GLOBAL_RETRIEVER is not None:
        return
        
    pdf_path = "customerTalk.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found in the directory.")
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    GLOBAL_RETRIEVER = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    
    GLOBAL_METADATA = {
        "filename": "customerTalk.pdf",
        "documents": len(docs),
        "chunks": len(chunks),
    }


def _get_retriever():
    """Fetch the global retriever, initializing it if necessary."""
    try:
        _init_global_retriever()
        return GLOBAL_RETRIEVER
    except FileNotFoundError:
        return None


# -------------------
# 3. Tools
# -------------------

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the configured local PDF (customerTalk.pdf).
    """
    retriever = _get_retriever()
    if retriever is None:
        return {
            "error": "The file customerTalk.pdf is missing from the server.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": GLOBAL_METADATA.get("filename"),
    }


tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    system_message = SystemMessage(
        content=(
            "You are a strict, closed-domain customer support assistant. "
            "Your ONLY source of knowledge is the data retrieved from the `rag_tool`. "
            "You MUST absolutely refuse to answer any questions outside of the document context. "
            "If the information to answer the user's question is NOT found in the data returned by `rag_tool`, "
            "you MUST reply ONLY with: 'not found relevent data contact to admin'. Do not try to answer using your own knowledge. "
            "CRITICAL WARNING: You must formulate your ENTIRE final response strictly in the EXACT SAME language that the user used to ask their question!"
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()
