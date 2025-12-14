import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
import torch

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(
    page_title="AI Chinese Essay Grader",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Fantastic UI"
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        font-family: "KaiTi", "SimKai", "Serif"; /* Traditional Chinese feel */
        font-size: 16px;
    }
    .score-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .feedback-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0068c9;
    }
    h1, h2, h3 {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. PIPELINE 1: SCORING MODEL (Runs Locally) ---
@st.cache_resource
def load_scoring_pipeline():
    """
    Loads the fine-tuned BERT model for scoring.
    Cached to prevent reloading on every interaction.
    """
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        # Load directly from Hub
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # Create a classification pipeline
        nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return nlp_pipeline
    except Exception as e:
        st.error(f"Error loading scoring model: {e}")
        return None

# --- 3. PIPELINE 2: FEEDBACK LLM (Uses Inference API) ---
def generate_feedback(essay, score, level, hf_token):
    """
    Calls Qwen2.5-3B-Instruct via Hugging Face Inference API.
    This ensures it runs fast on Streamlit Cloud without memory crashes.
    """
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    client = InferenceClient(model=model_id, token=hf_token)

    # Prompt Engineering for the "Decent Educator" Persona
    prompt = f"""
    Role: You are an experienced, supportive Chinese teacher (Traditional Chinese).
    Task: Provide feedback for a Primary Student's narrative essay.
    
    Student's Essay:
    "{essay}"
    
    Grading Results:
    - Score: {score}/100
    - Proficiency Level: {level}
    
    Instructions:
    1. Tone: Encouraging but educational. Be lenient but point out 1 specific area for improvement.
    2. Format:
       - 🌟 **Strengths**: Mention 1-2 good things (creativity, vocab, or structure).
       - 💡 **Suggestion**: Give 1 actionable tip to improve 'Show, Don't Tell'.
       - 📝 **Correction**: Fix one sentence or vocabulary misuse if found.
    3. Language: Traditional Chinese (Cantonese context aware if applicable).
    4. Length: Keep it concise (under 200 words).
    """

    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages, max_tokens=500, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {e}"

# --- 4. MAIN APP LOGIC ---

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/classroom.png", width=80)
        st.title("Teacher's Desk")
        st.info("This app uses a Two-Stage Pipeline:")
        st.markdown("1. **BERT Classifier**: Predicts the score.")
        st.markdown("2. **Qwen2.5-3B**: Generates personalized feedback.")
        
        # Get Token from Secrets
        if 'HF_TOKEN' in st.secrets:
            hf_token = st.secrets['HF_TOKEN']
            st.success("Hugging Face Token Loaded ✅")
        else:
            hf_token = st.text_input("Enter Hugging Face Token:", type="password")
            if not hf_token:
                st.warning("Please enter your token to generate feedback.")

    # Main Content
    st.title("📝 AI Chinese Essay Grader")
    st.markdown("Paste a primary school Chinese essay below to get an instant score and feedback.")

    essay_text = st.text_area("Student Essay Input", height=300, placeholder="在此輸入作文...")

    if st.button("Grade Essay", type="primary", use_container_width=True):
        if not essay_text.strip():
            st.warning("Please enter an essay first!")
            return
        
        if not hf_token:
            st.error("Hugging Face Token is missing!")
            return

        # Progress bar for UX
        progress_text = "Analyzing text structure..."
        my_bar = st.progress(0, text=progress_text)

        # --- EXECUTE PIPELINE 1 ---
        scorer = load_scoring_pipeline()
        if scorer:
            my_bar.progress(30, text="Calculating proficiency score...")
            
            # Since the model is a classifier, we need to map the label to a score/level.
            # Assuming the model outputs labels like "LABEL_0", "LABEL_1", etc.
            # You might need to adjust this mapping based on your specific training labels.
            # For this demo, we simulate the score extraction from the label or raw logits.
            
            result = scorer(essay_text[:512]) # Truncate to 512 for BERT
            label = result[0]['label']
            confidence = result[0]['score']
            
            # MAPPING LOGIC (Adjust based on your specific model's label map)
            # Example: If your model outputs "Excellent", "Good", "Needs Improvement"
            # Or if it outputs LABEL_0 (Low), LABEL_1 (Mid), LABEL_2 (High)
            
            # Placeholder mapping for demonstration:
            if "LABEL_2" in label or "Excellent" in label:
                pred_level = "Excellent"
                pred_score = int(85 + (confidence * 10)) # Dynamic score
            elif "LABEL_1" in label or "Good" in label:
                pred_level = "Good"
                pred_score = int(60 + (confidence * 20))
            else: # LABEL_0 or Needs Improvement
                pred_level = "Needs Improvement"
                pred_score = int(40 + (confidence * 15))
            
            # Cap score at 100
            pred_score = min(100, pred_score)

            my_bar.progress(60, text="Generating teacher feedback with Qwen2.5...")

            # --- EXECUTE PIPELINE 2 ---
            feedback = generate_feedback(essay_text, pred_score, pred_level, hf_token)
            
            my_bar.progress(100, text="Done!")
            my_bar.empty()

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="score-card">
                    <h3 style="margin:0; color:#888;">Proficiency</h3>
                    <h1 style="font-size: 48px; color: #0068c9; margin: 10px 0;">{pred_level}</h1>
                    <h2 style="color: #333;">{pred_score}/100</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 👩‍🏫 Teacher's Feedback")
                st.markdown(f"""
                <div class="feedback-box">
                    {feedback}
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
