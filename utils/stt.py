"""
Speech Recognition Module using Whisper Large-v3
Handles audio preprocessing and transcription
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment

def transcribe_audio(audio_path):
    """
    Convert audio file to text using Whisper ASR model
    Args:
        audio_path: Path to input audio file
    Returns:
        Transcribed English text
    """
    # Configure hardware settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to proper audio format
    audio = AudioSegment.from_file(audio_path)
    processed_audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = audio_path.replace(".mp3", ".wav")
    processed_audio.export(wav_path, format="wav")
    
    # Initialize ASR model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    
    # Process audio input
    inputs = processor(
        wav_path, 
        sampling_rate=16000,
        return_tensors="pt",
        truncation=True,
        chunk_length_s=30,
        stride_length_s=5
    ).to(device)
    
    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(**inputs, language="en", task="transcribe")
    
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]