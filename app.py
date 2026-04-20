import os
import uuid
import base64

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from audio_recorder_streamlit import audio_recorder
from voice_handler import convert_audio_to_text, convert_text_to_audio

from langgraph_backend import (
    chatbot,
    retrieve_all_threads,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Main Layout ========================
st.title("Customer Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Add the audio recorder at the bottom before chat input
st.markdown("---")
# Centering the mic logic so the component iframe doesn't stretch and turn white
col1, col2 = st.columns([1, 15])
with col1:
    audio_bytes = audio_recorder(text="", icon_size="2x")

user_input = st.chat_input("Ask about your document or use tools")

if "last_audio_bytes" not in st.session_state:
    st.session_state["last_audio_bytes"] = None

processed_audio_text = None
if audio_bytes and audio_bytes != st.session_state["last_audio_bytes"]:
    st.session_state["last_audio_bytes"] = audio_bytes
    with st.spinner("🎙️ Transcribing voice..."):
        processed_audio_text = convert_audio_to_text(audio_bytes)

final_input = user_input or processed_audio_text

if final_input:
    st.session_state["message_history"].append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.text(final_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=final_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    if isinstance(message_chunk.content, str):
                        yield message_chunk.content
                    elif isinstance(message_chunk.content, list):
                        for block in message_chunk.content:
                            if isinstance(block, str):
                                yield block
                            elif isinstance(block, dict) and "text" in block:
                                yield block["text"]

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Autoplay AI voice if the user used the microphone
    if processed_audio_text:
        audio_path = convert_text_to_audio(ai_message)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                # Create an invisible autoplay audio element
                md = f"""
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
            # Cleanup temp mp3
            try:
                os.remove(audio_path)
            except Exception:
                pass

st.divider()