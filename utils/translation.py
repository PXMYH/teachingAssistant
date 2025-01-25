"""
Text Translation Module using NLLB-3.3B model
Handles text segmentation and batch translation
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_text(text):
    """
    Translate English text to Simplified Chinese
    Args:
        text: Input English text
    Returns:
        Translated Chinese text
    """
    # Initialize translation model
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    
    # Split long text into manageable chunks
    max_chunk_length = 1000
    text_chunks = [
        text[i:i+max_chunk_length] 
        for i in range(0, len(text), max_chunk_length)
    ]
    
    translated_chunks = []
    for chunk in text_chunks:
        # Prepare model inputs
        inputs = tokenizer(
            chunk, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"],
            max_new_tokens=1024
        )
        translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return "".join(translated_chunks)