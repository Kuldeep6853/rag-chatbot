from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
hf_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=2048,
    do_sample=False,
    repetition_penalty=1.03,
)
llm = ChatHuggingFace(llm=hf_endpoint)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# -------------------
# 2. JSON retriever store (Global)
# -------------------
GLOBAL_RETRIEVER = None
GLOBAL_METADATA = {}


def _init_global_retriever():
    global GLOBAL_RETRIEVER, GLOBAL_METADATA
    if GLOBAL_RETRIEVER is not None:
        return
        
    json_path = "framerScheme.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found in the directory.")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in data]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    GLOBAL_RETRIEVER = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    
    GLOBAL_METADATA = {
        "filename": "framerScheme.json",
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
    Retrieve relevant information from the configured local JSON (framerScheme.json).
    """
    retriever = _get_retriever()
    if retriever is None:
        return {
            "error": "The file framerScheme.json is missing from the server.",
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
            "You are a strict, closed-domain agricultural customer support assistant. "
            "Your ONLY source of knowledge is the data retrieved from the `rag_tool`. "
            "You MUST absolutely refuse to answer any questions outside of the document context. "
            "WARNING: Even if you know the answer to a general knowledge question (e.g. 'what is the capital of Delhi', 'who is the president', etc.), you MUST refuse to answer it unless the answer is explicitly written in the retrieved document context. "
            "If the exact information to answer the user's question is NOT found in the data returned by `rag_tool`, "
            "you MUST reply ONLY with exactly: 'not found relevent data contact to admin'. Do not add any other words or try to answer using your own knowledge. "
            "CRITICAL WARNING: You must formulate your ENTIRE final response strictly in the EXACT SAME language that the user used to ask their question! "
            "IMPORTANT POLICY: If the user's query is related to crop diseases, prevention, or medicine/pesticides, you MUST always append a disclaimer at the end of your response advising them to consult an expert. You MUST translate this disclaimer into the EXACT SAME language as the user's query (e.g., if the user asks in Hindi, the disclaimer MUST be in Hindi, such as 'नोट: उपयोग करने से पहले, कृपया किसी कृषि विशेषज्ञ से पूछ लें।'). The English version of this disclaimer is: 'Note: Before use, please ask any agricultural expert'."
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
