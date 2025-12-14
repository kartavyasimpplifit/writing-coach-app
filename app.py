import streamlit as st
import os

# --- 1. CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="AI Essay Grader",
    page_icon="📝",
    layout="wide"
)

# --- 2. LAZY IMPORTS (Prevents startup crash) ---
# We verify libraries are installed but don't load them into RAM yet
try:
    import torch
    from huggingface_hub import InferenceClient
except ImportError:
    st.error("Libraries missing! Did you upload requirements.txt?")
    st.stop()

# --- 3. UI SETUP ---
st.title("📝 AI Chinese Essay Grader")
st.markdown("Paste your essay below. The model will load only when you click 'Grade'.")

# Check for Token
if 'HF_TOKEN' in st.secrets:
    hf_token = st.secrets['HF_TOKEN']
else:
    hf_token = st.text_input("Enter Hugging Face Token:", type="password")

essay_text = st.text_area("Student Essay Input", height=250)

# --- 4. SCORING FUNCTION (Cached) ---
@st.cache_resource
def get_scorer():
    """
    Imports and loads model ONLY when needed.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    
    print("Loading Scoring Model...") # Check your logs for this
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # We define the pipeline
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# --- 5. MAIN LOGIC ---
if st.button("Grade Essay", type="primary"):
    if not essay_text.strip():
        st.warning("Please enter an essay!")
        st.stop()
    
    if not hf_token:
        st.error("Missing HF Token!")
        st.stop()

    # Progress bar
    progress_bar = st.progress(0, text="Waking up the AI...")

    try:
        # Load heavy model NOW (saves startup time)
        scorer = get_scorer()
        progress_bar.progress(40, text="Scoring essay...")
        
        # 1. Pipeline 1: Scoring
        # Truncate to 512 to prevent BERT crash
        result = scorer(essay_text, truncation=True, max_length=512)
        label = result[0]['label']
        score_raw = result[0]['score']
        
        # Map labels
        if "Excellent" in label or "LABEL_2" in label:
            pred_level = "Excellent"
            pred_score = int(85 + (score_raw * 10))
        elif "Good" in label or "LABEL_1" in label:
            pred_level = "Good"
            pred_score = int(60 + (score_raw * 20))
        else:
            pred_level = "Needs Improvement"
            pred_score = int(40 + (score_raw * 15))
        
        pred_score = min(100, pred_score) # Cap at 100

        progress_bar.progress(70, text="Generating feedback...")

        # 2. Pipeline 2: Feedback (API)
        client = InferenceClient(model="Qwen/Qwen2.5-3B-Instruct", token=hf_token)
        prompt = f"Act as a strict Chinese teacher. Grade this essay: {essay_text}. Score: {pred_score}. Level: {pred_level}. Give 1 specific improvement in Traditional Chinese."
        
        feedback_response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=300
        )
        feedback = feedback_response.choices[0].message.content

        progress_bar.progress(100, text="Done!")
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Proficiency", value=pred_level)
            st.metric(label="Score", value=f"{pred_score}/100")
        with col2:
            st.info(feedback)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
