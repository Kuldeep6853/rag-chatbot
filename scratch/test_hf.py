import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

@tool
def sample_tool(query: str) -> dict:
    """This tool does something."""
    return {"query": query, "result": "sample"}

try:
    print("Initializing HuggingFaceEndpoint...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
    )
    print("Initializing ChatHuggingFace...")
    chat_model = ChatHuggingFace(llm=llm)
    print("Binding tools...")
    llm_with_tools = chat_model.bind_tools([sample_tool])
    print("Invoking...")
    res = llm_with_tools.invoke("Use the sample tool to find the weather")
    print(res)
    print(res.tool_calls)
except Exception as e:
    print(f"Error: {e}")
