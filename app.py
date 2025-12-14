from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# PIPELINE 2: FEEDBACK GENERATOR (LLM)
# ==========================================
@st.cache_resource
def load_feedback_model():
    """
    Model: Qwen/Qwen2.5-0.5B-Instruct
    Why: It is the best 'Tiny' LLM for Chinese. 
    It fits in Streamlit Cloud's free memory limit.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Feedback Model: {e}")
        return None, None

feedback_tokenizer, feedback_model = load_feedback_model()

def generate_smart_feedback(essay_text, score, category):
    """
    Uses the LLM to read the essay and write custom feedback.
    """
    # 1. Define the Teacher Persona & Prompt
    # We feed the score/category to the LLM so its feedback matches the grade.
    prompt = f"""
    你是香港中學中文老師。請閱讀以下學生作文，並根據評分提供評語。
    
    【作文資訊】
    分數：{score}/100
    等級：{category} (Excellent/Good/Needs Improvement)
    
    【學生作文】
    {essay_text}
    
    【任務】
    請提供以下兩部分的評語 (用繁體中文)：
    1. **關鍵見解 (Key Insights)**: 簡單評論文章的優點 (例如：修辭、結構、情感)。
    2. **改進建議 (Key Improvement Areas)**: 給出 3 個具體可行的建議點 (Bullet points)。
    
    語氣：鼓勵性、專業。
    """

    # 2. Format Input for Qwen
    messages = [
        {"role": "system", "content": "你是一位專業、親切的中文寫作教練。"},
        {"role": "user", "content": prompt}
    ]
    text = feedback_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. Generate
    model_inputs = feedback_tokenizer([text], return_tensors="pt").to(feedback_model.device)
    
    generated_ids = feedback_model.generate(
        model_inputs.input_ids,
        max_new_tokens=300,  # Limit length to keep it fast
        temperature=0.7,     # Creativity balance
        do_sample=True
    )
    
    # 4. Decode Output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = feedback_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response
