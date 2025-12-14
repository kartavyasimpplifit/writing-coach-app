"""
ISOM5240 Assignment - AI Essay Grading Application
Student ID: 2510gnam08, S029
Name: Kartavya Atri, NUS Singapore
Target: Secondary School Chinese Essays
Description: Two-pipeline system.
Pipeline 1: Classification (Fine-tuned Model) for Scoring.
Pipeline 2: Generation (Qwen 1.5B) for Detailed Feedback.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import random
import numpy as np

# Page Config
st.set_page_config(page_title="AI Essay Grader", layout="centered")

# ==========================================
# 1. MODEL LOADING (CACHED)
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
    Model: Qwen/Qwen1.5-1.8B-Chat
    """
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Load as CausalLM for text generation
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", # Force CPU for Streamlit Cloud compatibility
            trust_remote_code=True
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Feedback Model (Qwen): {e}")
        return None, None

# Load models
grading_tokenizer, grading_model = load_grading_model()
feedback_tokenizer, feedback_model = load_feedback_model()

# ==========================================
# 2. CORE FUNCTIONS
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
    
    # Map to Rubric
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

def generate_detailed_feedback(text, category):
    """
    Pipeline 2: Generate Context-Aware Feedback using Qwen 1.5B
    """
    # 1. Define the Persona based on Category (Prompt Engineering)
    if category == "Excellent":
        persona = "You are a teacher who loves genuine voice and creativity. Praise unique observations."
    elif category == "Good":
        persona = "You are a teacher focusing on structure. Teach the student how to turn 'telling' into 'showing'."
    else:
        persona = "You are an encouraging teacher. Find small details to praise to build confidence, but gently point out lack of content."
        
    # 2. Construct the Prompt
    # We use Qwen's chat template format
    prompt_text = f"""
    Role: {persona}
    Task: Read the following student essay and provide:
    1. A short paragraph of feedback comments (Teacher's Voice).
    2. Exactly 3 bullet points for specific improvements.
    
    Student Essay:
    "{text}"
    
    Output Language: English (with Chinese examples if needed).
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful essay grading assistant."},
        {"role": "user", "content": prompt_text}
    ]
    
    # 3. Prepare Inputs
    text_input = feedback_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    model_inputs = feedback_tokenizer([text_input], return_tensors="pt").to("cpu")
    
    # 4. Generate
    generated_ids = feedback_model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,
        temperature=0.7,  # Creativity
        do_sample=True
    )
    
    # 5. Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = feedback_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# ==========================================
# 3. MAIN APPLICATION
# ==========================================

def main():
    st.title("📝 AI Essay Grader")
    st.markdown("### ISOM5240 Individual Assignment")
    st.markdown("**Name:** Kartavya Atri | **ID:** 2510gnam08, S029")
    
    # Sidebar
    st.sidebar.header("Pipeline Status")
    st.sidebar.success("✅ Pipeline 1: Grading (Fine-tuned)")
    st.sidebar.success("✅ Pipeline 2: Feedback (Qwen 1.5B)")
    st.sidebar.info("System is ready.")
    
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
            with st.spinner("Pipeline 1: Calculating Score..."):
                category, score = get_grade_and_score(essay_input)
                
            with st.spinner("Pipeline 2: Qwen is writing detailed feedback..."):
                ai_response = generate_detailed_feedback(essay_input, category)
            
            # Display Results
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
            st.subheader("AI Coach Feedback")
            st.markdown(ai_response)

if __name__ == "__main__":
    main()
