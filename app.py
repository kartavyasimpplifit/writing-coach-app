import streamlit as st
import traceback

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI Essay Grader",
    page_icon="📝",
    layout="wide"
)

# --- 2. LAZY IMPORTS ---
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import InferenceClient
except ImportError as e:
    st.error(f"Libraries missing! {e}")
    st.stop()

# --- 3. UI SETUP ---
st.title("📝 AI Chinese Essay Grader")
st.markdown("Paste your essay below.")

# Check for Token
if 'HF_TOKEN' in st.secrets:
    hf_token = st.secrets['HF_TOKEN']
else:
    hf_token = st.text_input("Enter Hugging Face Token:", type="password")

essay_text = st.text_area("Student Essay Input", height=250)

# --- 4. SCORING FUNCTION ---
@st.cache_resource
def get_scorer():
    print("Loading Scoring Model...")
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    
    # Load Tokenizer & Model explicitly
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # Force CPU (device=-1) to prevent memory crashes on Cloud
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

# --- 5. MAIN LOGIC ---
if st.button("Grade Essay", type="primary"):
    if not essay_text.strip():
        st.warning("Please enter an essay!")
        st.stop()
    
    if not hf_token:
        st.error("Missing HF Token! Please add it to secrets.toml or input box.")
        st.stop()

    # Progress bar
    progress_bar = st.progress(0, text="Starting AI engines...")

    try:
        # --- PIPELINE 1: SCORING ---
        scorer = get_scorer()
        progress_bar.progress(30, text="Analyzing text structure...")
        
        # [CRITICAL FIX] Explicitly handle long text truncation
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

        progress_bar.progress(60, text="Teacher is writing feedback...")

        # --- PIPELINE 2: FEEDBACK ---
        # Switch to 72B model which is supported by the Serverless API
        # If this fails, try "google/gemma-2-9b-it"
        client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
        
        prompt = f"""
        Role: Strict but fair Chinese Teacher (Traditional Chinese).
        Task: Provide feedback for this primary school essay.
        Essay: "{essay_text[:1000]}"
        Grading: {pred_score}/100 ({pred_level})
        
        Requirements:
        1. Identify 1 Strength.
        2. Identify 1 Specific Area for Improvement (Vocabulary or Structure).
        3. Keep it encouraging but concise (under 150 words).
        """
        
        # Using chat_completion which auto-formats the prompt
        feedback_response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=350,
            temperature=0.7
        )
        feedback = feedback_response.choices[0].message.content

        progress_bar.progress(100, text="Done!")
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Proficiency", value=pred_level)
            st.metric(label="Score", value=f"{pred_score}/100")
        with col2:
            st.success("Teacher's Feedback:")
            st.write(feedback)

    except Exception as e:
        st.error("❌ An error occurred during processing.")
        st.write("Debug info:")
        st.code(traceback.format_exc())
