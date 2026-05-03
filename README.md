# LangGraph Agricultural Chatbot

This project is a multilingual, voice-enabled AI support system designed for the agricultural sector. 

## Key Highlights (PPT Points)

1. **Voice-Enabled Multilingual Support:** Features integrated Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities using Google Speech Recognition and gTTS, heavily optimized for Hindi and other regional languages.
2. **Advanced Agentic Architecture:** Built using LangGraph to manage complex conversation flows and tool execution via a structured state graph, providing reliable and predictable agent behavior.
3. **Powerful Hugging Face LLM:** Powered by the cutting-edge `Qwen/Qwen2.5-72B-Instruct` model through Hugging Face Endpoints, delivering high-quality, reasoning-based text generation.
4. **Robust RAG Pipeline:** Implements Retrieval-Augmented Generation (RAG) using FAISS and `paraphrase-multilingual-MiniLM` embeddings to fetch highly relevant context from a local JSON dataset (`framerScheme.json`).
5. **Interactive Streamlit UI:** Features a responsive, chat-based frontend developed with Streamlit, complete with real-time processing locks, chat history, and embedded audio playback.
6. **Strict Domain Guardrails:** Enforces rigorous system prompt constraints, ensuring the AI strictly answers from the provided dataset and actively refuses out-of-domain questions to prevent hallucinations.
7. **Automated Safety Disclaimers:** Automatically detects when a user asks about crop diseases or pesticides and appends expert consultation warnings in the exact language the user queried in.
8. **Stateless & Cloud-Ready Design:** Utilizes Streamlit session states and on-the-fly vector store initialization, avoiding database dependencies and making it perfect for direct deployment to Streamlit Cloud.
