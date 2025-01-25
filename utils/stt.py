import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment

def transcribe_audio(audio_path):
    """使用Whisper Large-v3进行语音识别"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 转换音频格式
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = audio_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    
    # 加载模型
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    
    # 处理音频
    inputs = processor(
        wav_path, 
        sampling_rate=16000,
        return_tensors="pt",
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, language="en", task="transcribe")
    
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]