from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_text(text):
    """使用NLLB-3.3B进行翻译"""
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    
    # 分块处理长文本
    max_chunk_length = 1000
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"],
            max_new_tokens=1024
        )
        translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return "".join(translated_chunks)