"""
Main entry point for the Audio Translation Web Application
Handles file upload, processing pipeline, and UI rendering
"""

import streamlit as st
import os
import time
# from dotenv import load_dotenv
from utils.stt import transcribe_audio
from utils.translation import translate_text
from utils.tts import generate_speech

# Initialize environment configurations
# load_dotenv()
os.makedirs("temp/uploads", exist_ok=True)
os.makedirs("temp/outputs", exist_ok=True)

def configure_page():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title="Audio Translator",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
            .reportview-container {margin-top: -2em;}
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
        </style>
    """, unsafe_allow_html=True)

def handle_file_processing(upload_path):
    """
    Execute the complete processing pipeline:
    1. Speech-to-Text (STT)
    2. Machine Translation
    3. Text-to-Speech (TTS)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # STT Phase
        status_text.markdown("🔍 **Performing Speech Recognition...**")
        english_text = transcribe_audio(upload_path)
        progress_bar.progress(30)
        
        # Translation Phase
        status_text.markdown("🌐 **Translating Content...**")
        chinese_text = translate_text(english_text)
        progress_bar.progress(60)
        
        # TTS Phase
        status_text.markdown("🎵 **Generating Chinese Speech...**")
        output_path = generate_speech(chinese_text)
        progress_bar.progress(100)
        
        # Display results
        status_text.success("✅ Processing Complete!")
        return english_text, chinese_text, output_path
        
    except Exception as e:
        status_text.error(f"❌ Processing Failed: {str(e)}")
        st.exception(e)
        raise

def render_results(english_text, chinese_text, output_path):
    """Display processing results in organized columns"""
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Recognition Results")
        st.code(english_text, language="text")
        
        st.subheader("Translation Results")
        st.code(chinese_text, language="text")

    with col2:
        st.subheader("Audio Output")
        st.audio(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Audio",
                data=f,
                file_name="translated_audio.wav",
                mime="audio/wav"
            )

def main():
    """Main application workflow"""
    configure_page()
    st.title("🎧 High-Quality Audio Translation System")
    st.markdown("Upload English Audio → Get Chinese Speech Output")

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Select Audio File (MP3/WAV)",
        type=["mp3", "wav"],
        accept_multiple_files=False
    )

    if uploaded_file:
        # Save uploaded file
        upload_path = os.path.join("temp/uploads", uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Execute processing pipeline
        results = handle_file_processing(upload_path)
        if results:
            render_results(*results)

if __name__ == "__main__":
    main()