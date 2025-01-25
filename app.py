import streamlit as st
import os
import time
from dotenv import load_dotenv
from utils.stt import transcribe_audio
from utils.translation import translate_text
from utils.tts import generate_speech

# 初始化环境
load_dotenv()
os.makedirs("temp/uploads", exist_ok=True)
os.makedirs("temp/outputs", exist_ok=True)

st.set_page_config(
    page_title="Audio Translator",
    page_icon="🎧",
    layout="wide"
)

def main():
    st.title("🎧 高精度音频翻译系统")
    st.markdown("上传英文音频 → 获取中文语音")

    # 文件上传区域
    uploaded_file = st.file_uploader(
        "选择音频文件 (MP3/WAV)",
        type=["mp3", "wav"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # 保存上传文件
        upload_path = os.path.join("temp/uploads", uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 处理进度可视化
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 语音识别
            status_text.markdown("🔍 **正在识别语音...**")
            english_text = transcribe_audio(upload_path)
            progress_bar.progress(30)
            
            # 文本翻译
            status_text.markdown("🌐 **正在翻译文本...**")
            chinese_text = translate_text(english_text)
            progress_bar.progress(60)
            
            # 语音合成
            status_text.markdown("🎵 **正在生成中文语音...**")
            output_path = generate_speech(chinese_text)
            progress_bar.progress(100)
            
            # 显示结果
            status_text.success("✅ 处理完成！")
            st.divider()
            
            # 结果展示列
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("识别结果")
                st.code(english_text, language="text")
                
                st.subheader("翻译结果")
                st.code(chinese_text, language="text")

            with col2:
                st.subheader("中文语音")
                st.audio(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="下载音频",
                        data=f,
                        file_name="translated_audio.wav",
                        mime="audio/wav"
                    )

        except Exception as e:
            status_text.error(f"❌ 处理失败: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()