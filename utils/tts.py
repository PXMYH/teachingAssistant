from TTS.api import TTS
import os

def generate_speech(text):
    """使用YourTTS生成中文语音"""
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=False,
        gpu=False
    )
    
    output_path = os.path.join("temp/outputs", f"output_{int(time.time())}.wav")
    
    # 使用示例参考语音（需自行准备或使用默认）
    ref_voice = "assets/reference_voice.wav" if os.path.exists("assets/reference_voice.wav") else None
    
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_voice,
        language="zh-cn",
        file_path=output_path
    )
    
    return output_path