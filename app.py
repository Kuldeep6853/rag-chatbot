import os
import uuid
import base64

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from audio_recorder_streamlit import audio_recorder
from voice_handler import convert_audio_to_text, convert_text_to_audio

from langgraph_backend import chatbot

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ============================ Main Layout ========================
st.title("Customer Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Add the audio recorder at the bottom before chat input (Hide if processing)
st.markdown("---")
col1, col2 = st.columns([1, 15])
with col1:
    if not st.session_state.is_processing:
        audio_bytes = audio_recorder(text="", icon_size="2x")
    else:
        audio_bytes = None

user_input = st.chat_input("Ask about your document or use tools", disabled=st.session_state.is_processing)

if "last_audio_bytes" not in st.session_state:
    st.session_state["last_audio_bytes"] = None

processed_audio_text = None
if audio_bytes and audio_bytes != st.session_state["last_audio_bytes"]:
    st.session_state["last_audio_bytes"] = audio_bytes
    with st.spinner("🎙️ Transcribing voice..."):
        processed_audio_text = convert_audio_to_text(audio_bytes)

final_input = user_input or processed_audio_text

if final_input and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state["message_history"].append({"role": "user", "content": final_input})
    st.rerun()

# If we are processing, generate the response, then unlock.
if st.session_state.is_processing:

    with st.chat_message("assistant"):
        def ai_only_stream():
            history = []
            for msg in st.session_state["message_history"]:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))

            for message_chunk, _ in chatbot.stream(
                {"messages": history},
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
    if st.session_state["last_audio_bytes"] is not None and user_input is None:
        audio_path = convert_text_to_audio(ai_message)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
            try:
                os.remove(audio_path)
            except Exception:
                pass

    st.session_state.is_processing = False
    st.rerun()

st.divider()