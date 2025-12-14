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

# Custom CSS for a professional UI
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        font-family: "KaiTi", "SimKai", "Serif";
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
    h1, h2, h3 { color: #333333; }
    </style>
""", unsafe_allow_html=True)

# --- 2. PIPELINE 1: SCORING MODEL (Runs Locally) ---
@st.cache_resource
def load_scoring_pipeline():
    """
    Loads the fine-tuned BERT model for scoring.
    Forces CPU (device=-1) to prevent memory crashes on Cloud.
    """
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # device=-1 ensures we use CPU, which is safer for Streamlit Cloud
        nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
        return nlp_pipeline
    except Exception as e:
        st.error(f"Error loading scoring model: {e}")
        return None

# --- 3. PIPELINE 2: FEEDBACK LLM (Uses Hugging Face API) ---
def generate_feedback(essay, score, level, hf_token):
    """
    Calls Qwen2.5-72B-Instruct via API. 
    Using 72B because smaller models often cause API routing errors.
    """
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    
    client = InferenceClient(model=model_id, token=hf_token)

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
        # Using chat_completion for better instruction following
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
        st.markdown("1. **BERT Classifier**: Predicts the score locally.")
        st.markdown("2. **Qwen-72B**: Generates feedback via API.")
        
        # Token Management
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
            st.error("Hugging Face Token is missing! Please add it to your secrets or the sidebar.")
            return

        # Progress bar
        progress_text = "Analyzing text structure..."
        my_bar = st.progress(0, text=progress_text)

        # --- EXECUTE PIPELINE 1 (SCORING) ---
        scorer = load_scoring_pipeline()
        if scorer:
            my_bar.progress(30, text="Calculating proficiency score...")
            
            try:
                # 1. Get RAW scores for ALL labels (top_k=None)
                predictions = scorer(essay_text, truncation=True, max_length=512, top_k=None)
                
                # --- [CRITICAL FIX: Added Chinese Labels] ---
                label_weights = {
                    "不及格": 45,       # Needs Improvement
                    "需改進": 45,       # Alternative for Needs Improvement
                    "良好": 75,         # Good
                    "優秀": 95,         # Excellent (Traditional)
                    "优秀": 95,         # Excellent (Simplified fallback)
                    "LABEL_0": 45,
                    "LABEL_1": 75,
                    "LABEL_2": 95
                }

                # 3. Calculate Weighted Score
                weighted_score = 0
                total_confidence = 0
                dominant_label = ""
                highest_conf = 0

                for p in predictions:
                    l = p['label']
                    c = p['score']
                    
                    if c > highest_conf:
                        highest_conf = c
                        dominant_label = l

                    if l in label_weights:
                        weighted_score += (label_weights[l] * c)
                        total_confidence += c
                
                # Final Score
                pred_score = int(weighted_score)
                
                # Fallback if dictionary lookup failed (e.g. if sum is 0)
                if pred_score == 0 and dominant_label:
                     # Attempt to salvage based on dominant label if it wasn't in dict
                     if "優" in dominant_label or "Exc" in dominant_label: pred_score = 90
                     elif "良" in dominant_label or "Good" in dominant_label: pred_score = 75
                     else: pred_score = 45

                # Determine Level based on the calculated score
                if pred_score >= 85:
                    pred_level = "Excellent (優秀)"
                elif pred_score >= 60:
                    pred_level = "Good (良好)"
                else:
                    pred_level = "Needs Improvement (需改進)"

                # 4. Debug Expander
                with st.expander("🔍 View Scoring Details (Debug)"):
                    st.write(f"Raw Output: {predictions}")
                    st.write(f"Weighted Calculation: {pred_score}")

            except Exception as e:
                st.error(f"Error during scoring: {e}")
                my_bar.empty()
                return

            my_bar.progress(60, text="Generating teacher feedback...")

            # --- EXECUTE PIPELINE 2 (FEEDBACK) ---
            feedback = generate_feedback(essay_text, pred_score, pred_level, hf_token)
            
            my_bar.progress(100, text="Done!")
            my_bar.empty()

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="score-card">
                    <h3 style="margin:0; color:#888;">Proficiency</h3>
                    <h1 style="font-size: 36px; color: #0068c9; margin: 10px 0;">{pred_level}</h1>
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
