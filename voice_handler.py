import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect

def convert_audio_to_text(audio_bytes: bytes) -> str:
    """Takes WAV audio bytes from the Streamlit frontend and returns the spoken text."""
    if not audio_bytes:
        return ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(audio_bytes)
        tmp_wav_path = tmp_wav.name

    text = ""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(tmp_wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="hi-IN")
    except sr.UnknownValueError:
        print("Error processing audio: No speech was detected (or the audio was empty).")
    except sr.RequestError as e:
        print(f"Error processing audio: Google API network request failed: {e}")
    except ValueError as e:
        print(f"Error processing audio: Audio format rejected (ValueError): {e}")
    except Exception as e:
        print(f"Error processing audio: Unknown {type(e).__name__} -> {e}")
    finally:
        try:
            os.remove(tmp_wav_path)
        except Exception:
            pass

    return text

def convert_text_to_audio(text: str) -> str:
    """Uses Google TTS to generate an MP3 audio file path from the given AI text."""
    if not text.strip():
        return ""
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3_path = tmp_mp3.name
        
    try:
        lang_code = detect(text)
    except Exception:
        lang_code = 'en'
        
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(tmp_mp3_path)
        return tmp_mp3_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return ""
