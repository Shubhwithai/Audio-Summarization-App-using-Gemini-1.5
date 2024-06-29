import streamlit as st
import tempfile
import os
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API for audio summarization
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
configure(api_key=GOOGLE_API_KEY)

def summarize_audio(audio_file_path):
    """Summarize the audio using Google's Generative AI."""
    model = GenerativeModel("gemini-1.5-pro-latest")
    with open(audio_file_path, "rb") as audio_file:
        response = model.generate_content([
            "Please summarize the following audio.",
            {"mime_type": "audio/wav", "data": audio_file.read()}
        ])
    return response.text

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.title('Audio Summarization App')

with st.expander("About this app"):
    st.write("""
        This app uses Google's Generative AI to summarize audio files. 
        Upload your audio file in WAV or MP3 format and get a concise summary of its content.
    """)

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])

if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    if audio_path:
        st.audio(audio_file)
        if st.button('Summarize Audio'):
            with st.spinner('Summarizing...'):
                try:
                    summary_text = summarize_audio(audio_path)
                    st.success("Summary generated successfully!")
                    st.info(summary_text)
                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")
    else:
        st.error("Failed to process the uploaded file.")
