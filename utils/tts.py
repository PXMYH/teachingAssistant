"""
Text-to-Speech Module using YourTTS
Handles speech synthesis and output generation
"""

from TTS.api import TTS
import os
import time

def generate_speech(text):
    """
    Convert Chinese text to natural-sounding speech
    Args:
        text: Input Chinese text
    Returns:
        Path to generated audio file
    """
    # Initialize TTS engine
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=False,
        gpu=False
    )
    
    # Create unique output filename
    output_path = os.path.join(
        "temp/outputs", 
        f"output_{int(time.time())}.wav"
    )
    
    # Use reference voice if available
    ref_voice = (
        "assets/reference_voice.wav" 
        if os.path.exists("assets/reference_voice.wav") 
        else None
    )
    
    # Generate speech output
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_voice,
        language="zh-cn",
        file_path=output_path
    )
    
    return output_path