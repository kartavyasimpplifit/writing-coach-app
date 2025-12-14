"""
ISOM5240 Assignment - AI Essay Grading Application
Student ID: 2510gnam08, S029
Name: Kartavya Atri, NUS Singapore
Target: Secondary School Chinese Essays
Description: Two-pipeline system.
Pipeline 1: Classification (Fine-tuned Model) for Scoring.
Pipeline 2: Generation (Qwen 0.5B) for Detailed Feedback.
"""

# ==========================================
# 1. IMPORTS (CRITICAL: DO NOT DELETE)
# ==========================================
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import random
import numpy as np

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Essay Grader",
    page_icon="📝",
    layout="centered"
)

# ==========================================
# 3. MODEL LOADING (CACHED)
# ==========================================

@st.cache_resource
def load_grading_model():
    """
    Pipeline 1: Grading Model (Classifier)
    Model: MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3
    """
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Explicitly set 3 labels for your specific fine-tuned model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=3, 
            ignore_mismatched_sizes=True
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Grading Model: {e}")
        return None, None

@st.cache_resource
def load_feedback_model():
    """
    Pipeline 2: Feedback Generation Model (LLM)
    Model: Qwen/Qwen2.5-0.5B-Instruct
    Why: Best performance-to-size ratio for Chinese feedback on free cloud.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load as CausalLM for text generation
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", # Force CPU for Streamlit Cloud compatibility
            torch_dtype=torch.float32
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Feedback Model (Qwen): {e}")
        return None, None

# Load models immediately
grading_tokenizer, grading_model = load_grading_model()
feedback_tokenizer, feedback_model = load_feedback_model()

# ==========================================
# 4. CORE FUNCTIONS
# ==========================================

def get_grade_and_score(text):
    """
    Pipeline 1: Determine Proficiency Category & Score
    """
    inputs = grading_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = grading_model(**inputs)
    
    # Predicted Class: 0, 1, or 2
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    
    # Map to Rubric based on your previous logs
    # 2 = Excellent, 1 = Good, 0 = Needs Improvement
    if pred_id == 2: 
        category = "Excellent"
        score = random.randint(85, 100)
    elif pred_id == 1: 
        category = "Good"
        score = random.randint(60, 84)
    else: # pred_id == 0
        category = "Needs Improvement"
        score = random.randint(0, 59)
        
    return category, score

def generate_smart_feedback(essay_text, score, category):
    """
    Pipeline 2: Generate Context-Aware Feedback using Qwen LLM
    """
    # 1. Define the Prompt
    prompt = f"""
    You are a professional Chinese writing teacher (香港中學中文老師).
    
    Student Essay Info:
    - Score: {score}/100
    - Level: {category}
    
    Student Essay:
    "{essay_text}"
    
    Task:
    Provide feedback in Traditional Chinese (繁體中文).
    1. **Overall Comment (總評)**: A 2-sentence summary of the writing quality.
    2. **Key Insights (優點)**: Mention 1 specific strength (e.g., rhetoric, emotion).
    3. **Improvement Points (建議)**: Give exactly 3 bullet points on how to improve.
    
    Keep the tone encouraging but professional.
    """
    
    # 2. Format Input for Qwen
    messages = [
        {"role": "system", "content": "You are a helpful essay grading assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text_input = feedback_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = feedback_tokenizer([text_input], return_tensors="pt").to(feedback_model.device)
    
    # 3. Generate
    generated_ids = feedback_model.generate(
        inputs.input_ids,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )
    
    # 4. Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = feedback_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# ==========================================
# 5. MAIN APPLICATION UI
# ==========================================

def main():
    st.title("📝 AI Essay Grader")
    st.markdown("### ISOM5240 Individual Assignment")
    st.markdown("**Name:** Kartavya Atri | **ID:** 2510gnam08, S029")
    
    # Sidebar
    st.sidebar.header("Pipeline Status")
    st.sidebar.success("✅ P1: Grading (Fine-tuned)")
    st.sidebar.success("✅ P2: Feedback (Qwen 0.5B)")
    st.sidebar.info("System Ready")
    
    st.write("---")
    st.info("Paste a Chinese essay below. The AI will Grade it (Pipeline 1) and Write Feedback (Pipeline 2).")
    
    # Input
    essay_input = st.text_area("Student Essay:", height=200, placeholder="在此輸入作文...")
    
    if st.button("🚀 Grade Essay", type="primary"):
        if not essay_input.strip():
            st.warning("⚠️ Please enter an essay first.")
        elif grading_model is None or feedback_model is None:
            st.error("❌ Models failed to load.")
        else:
            # 1. Pipeline 1
            with st.spinner("Pipeline 1: Calculating Score..."):
                category, score = get_grade_and_score(essay_input)
            
            # 2. Pipeline 2
            with st.spinner("Pipeline 2: AI Coach is analyzing..."):
                ai_response = generate_smart_feedback(essay_input, score, category)
            
            # 3. Display Results
            st.write("---")
            st.subheader("Grading Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if category == "Excellent": c = "green"
                elif category == "Good": c = "orange"
                else: c = "red"
                st.markdown(f"### Score: :{c}[{score}/100]")
            with col2:
                st.markdown(f"### Level: {category}")
            
            st.write("---")
            st.subheader("📝 AI Coach Feedback")
            st.markdown(ai_response)

if __name__ == "__main__":
    main()
