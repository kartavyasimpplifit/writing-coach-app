import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
import torch

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="AI Chinese Essay Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. AWESOME CUSTOM CSS ---
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');

    /* Global Styles */
    .main {
        background-color: #F8F9FA;
        font-family: 'Noto Sans TC', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 2px solid #E0E0E0;
        padding: 15px;
        font-size: 16px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #3498DB;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 25px;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.1s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        color: white;
    }

    /* Score Card Component */
    .score-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-top: 6px solid #2ECC71; /* Default Green */
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .score-value {
        font-size: 5rem;
        font-weight: 800;
        color: #2C3E50;
        line-height: 1;
        margin: 10px 0;
    }
    .score-label {
        font-size: 1.5rem;
        font-weight: 600;
        color: #27AE60;
        background-color: #E8F8F5;
        padding: 5px 20px;
        border-radius: 50px;
        display: inline-block;
    }

    /* Feedback Box Component */
    .feedback-box {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-left: 6px solid #F1C40F;
        height: 100%;
    }
    .feedback-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .feedback-content {
        font-size: 1.1rem;
        color: #555;
        line-height: 1.6;
    }
    
    /* Loading Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3498DB, #2ECC71);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---

@st.cache_resource
def load_scoring_pipeline():
    """Load BERT model efficiently."""
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # Force CPU to prevent Cloud crashes
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"⚠️ Model Loading Error: {e}")
        return None

def generate_feedback(essay, score, level, hf_token):
    """Generate feedback using Qwen-72B via API."""
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    
    if not hf_token:
        return "⚠️ Please provide a Hugging Face Token to see feedback."

    client = InferenceClient(model=model_id, token=hf_token)

    prompt = f"""
    You are an experienced, encouraging Chinese teacher (Traditional Chinese).
    
    **Task:** Grade this Primary Student's essay.
    **Student Essay:** "{essay}"
    **Calculated Score:** {score}/100 ({level})
    
    **Instructions:**
    1. **Tone:** Warm but professional. 
    2. **Structure:**
       - 🌟 **亮點 (Highlights):** 1-2 sentences on what they did well.
       - 💡 **建議 (Suggestions):** 1 specific tip to improve 'Show, Don't Tell'.
       - ✍️ **修訂 (Correction):** Correct one sentence or phrase for better vocabulary.
    3. **Language:** Traditional Chinese (繁體中文).
    4. **Length:** Keep it concise (~150 words).
    """

    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages, max_tokens=600, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Feedback Error: {str(e)}"

# --- 4. MAIN APP LAYOUT ---

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3429/3429414.png", width=80)
        st.markdown("## 👩‍🏫 Teacher's Desk")
        st.markdown("---")
        st.info("**System Status:**\n\n🟢 Scoring Model (Ready)\n\n🟢 Feedback AI (Ready)")
        
        st.markdown("### 🔑 API Settings")
        # Token Handling
        if 'HF_TOKEN' in st.secrets:
            hf_token = st.secrets['HF_TOKEN']
            st.success("Token Loaded from Secrets 🔒")
        else:
            hf_token = st.text_input("Hugging Face Token", type="password", help="Paste your HF Write Token here.")
            if not hf_token:
                st.warning("Token required for feedback.")
        
        st.markdown("---")
        st.caption("v2.0 | Powered by Qwen & BERT")

    # --- Header ---
    st.markdown('<div class="main-header">🎓 AI Chinese Essay Grader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Instant Scoring & Personalized Feedback for Hong Kong Primary Students</div>', unsafe_allow_html=True)

    # --- Input Section ---
    col_input, col_space = st.columns([1, 0.05]) # Just styling spacing
    with col_input:
        essay_text = st.text_area("📝 Paste Student Essay Here:", height=250, placeholder="在此輸入作文...")
        
        grade_btn = st.button("✨ Grade Essay Now", type="primary")

    # --- Logic & Output ---
    if grade_btn:
        if not essay_text.strip():
            st.toast("⚠️ Please enter an essay first!", icon="🚫")
            return

        # Progress UI
        progress_text = st.empty()
        my_bar = st.progress(0)

        # 1. Scoring Phase
        scorer = load_scoring_pipeline()
        if scorer:
            progress_text.text("🧠 Analyzing text structure and vocabulary...")
            my_bar.progress(30)
            
            try:
                # Get raw scores
                predictions = scorer(essay_text, truncation=True, max_length=512, top_k=None)
                
                # Weighted Scoring Logic
                label_weights = {
                    "不及格": 45, "Needs Improvement": 45, "LABEL_0": 45,
                    "良好": 75, "Good": 75, "LABEL_1": 75,
                    "優秀": 95, "Excellent": 95, "LABEL_2": 95
                }

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
                    # Fuzzy match keys
                    for key, val in label_weights.items():
                        if key in l:
                            weighted_score += (val * c)
                            total_confidence += c
                            break
                
                pred_score = int(weighted_score)
                
                # Level Determination
                if pred_score >= 85:
                    pred_level = "Excellent (優秀)"
                    color_theme = "#2ECC71" # Green
                    bg_theme = "#E8F8F5"
                elif pred_score >= 60:
                    pred_level = "Good (良好)"
                    color_theme = "#F39C12" # Orange
                    bg_theme = "#FEF9E7"
                else:
                    pred_level = "Needs Improvement (需改進)"
                    color_theme = "#E74C3C" # Red
                    bg_theme = "#FDEDEC"

            except Exception as e:
                st.error(f"Scoring Failed: {e}")
                my_bar.empty()
                return

            # 2. Feedback Phase
            progress_text.text("✍️ Teacher is writing comments...")
            my_bar.progress(70)
            
            feedback = generate_feedback(essay_text, pred_score, pred_level, hf_token)
            
            my_bar.progress(100)
            progress_text.empty()
            my_bar.empty()

            # --- Results Display ---
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2], gap="large")

            with res_col1:
                # Dynamic CSS for Score Card based on result
                st.markdown(f"""
                <div class="score-card" style="border-top: 6px solid {color_theme};">
                    <h3 style="margin:0; color:#888; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 1px;">Proficiency Score</h3>
                    <div class="score-value">{pred_score}</div>
                    <div class="score-label" style="color: {color_theme}; background-color: {bg_theme};">{pred_level}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="feedback-box">
                    <div class="feedback-title">
                        👩‍🏫 Teacher's Feedback
                    </div>
                    <div class="feedback-content">
                        {feedback}
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
