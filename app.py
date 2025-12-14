import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
import torch

# ==========================================
# PART 1: BACKEND LOGIC (AI & SCORING)
# ==========================================

@st.cache_resource
def load_scoring_pipeline():
    """
    Loads the fine-tuned BERT model for scoring.
    Forces CPU (device=-1) to prevent memory crashes on Streamlit Cloud.
    """
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # device=-1 ensures CPU usage
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"⚠️ Scoring Model Error: {e}")
        return None

def calculate_weighted_score(predictions):
    """
    Calculates a precise 0-100 score based on the probability of all labels.
    """
    # Define score mapping for English and Chinese labels
    label_weights = {
        "不及格": 45, "Needs Improvement": 45, "LABEL_0": 45,
        "良好": 75, "Good": 75, "LABEL_1": 75,
        "優秀": 95, "Excellent": 95, "LABEL_2": 95
    }

    weighted_score = 0
    total_confidence = 0
    
    for p in predictions:
        l = p['label']
        c = p['score']
        
        # Fuzzy match the label key
        for key, val in label_weights.items():
            if key in l:
                weighted_score += (val * c)
                total_confidence += c
                break
    
    final_score = int(weighted_score)
    
    # Determine Level Label
    if final_score >= 85:
        level = "Excellent (優秀)"
        color = "#2ECC71" # Green
    elif final_score >= 60:
        level = "Good (良好)"
        color = "#F39C12" # Orange
    else:
        level = "Needs Improvement (需改進)"
        color = "#E74C3C" # Red
        
    return final_score, level, color

def generate_feedback(essay, score, level, hf_token):
    """
    Generates qualitative feedback using Qwen2.5-72B via API.
    """
    if not hf_token:
        return "⚠️ Token missing. Please add your Hugging Face token."

    # Use 72B model for best Chinese performance via API
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    client = InferenceClient(model=model_id, token=hf_token)

    prompt = f"""
    You are an experienced Chinese teacher (Traditional Chinese).
    
    **Task:** Grade this Primary Student's essay.
    **Student Essay:** "{essay}"
    **Calculated Score:** {score}/100 ({level})
    
    **Instructions:**
    1. **Tone:** Encouraging but educational.
    2. **Structure:**
       - 🌟 **Highlights:** 1 sentence on what is good.
       - 💡 **Suggestion:** 1 specific tip to improve 'Show, Don't Tell'.
       - ✍️ **Correction:** Correct one specific vocabulary mistake.
    3. **Language:** Traditional Chinese (繁體中文).
    4. **Length:** Keep it concise (under 150 words).
    """

    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages, max_tokens=500, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Feedback Error: {str(e)}"


# ==========================================
# PART 2: FRONTEND UI (LAYOUT & CSS)
# ==========================================

def main():
    # 1. Page Config
    st.set_page_config(
        page_title="AI Chinese Essay Grader",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Custom CSS (Fixed White Input Bug)
    st.markdown("""
        <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');

        .main {
            background-color: #F8F9FA;
            font-family: 'Noto Sans TC', sans-serif;
        }
        
        /* FIX: Force Input Text Color to Dark Grey */
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #333333 !important; /* Fixes white-on-white text */
            caret-color: #333333; /* Fixes invisible cursor */
            border-radius: 12px;
            border: 2px solid #E0E0E0;
            padding: 15px;
            font-size: 16px;
            font-family: "KaiTi", "SimKai", "Serif";
        }
        .stTextArea textarea:focus {
            border-color: #3498DB;
            box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
        }
        /* Fix placeholder color */
        .stTextArea textarea::placeholder {
            color: #888888;
        }

        /* Header Styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2C3E50;
            text-align: center;
            margin-top: -20px;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #7F8C8D;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Score Card */
        .score-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .score-value {
            font-size: 4.5rem;
            font-weight: 800;
            color: #2C3E50;
            margin: 10px 0;
            line-height: 1;
        }
        .score-label {
            font-size: 1.2rem;
            font-weight: 600;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
        }

        /* Feedback Box */
        .feedback-box {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-left: 5px solid #3498DB;
            height: 100%;
        }
        
        /* Button */
        .stButton>button {
            width: 100%;
            border-radius: 30px;
            background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
            color: white;
            border: none;
            padding: 12px 0;
            font-weight: bold;
            font-size: 18px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52,152,219,0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Sidebar Layout
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3429/3429414.png", width=70)
        st.markdown("## 👩‍🏫 Teacher's Desk")
        
        # Token Input
        if 'HF_TOKEN' in st.secrets:
            hf_token = st.secrets['HF_TOKEN']
            st.success("API Token Loaded 🔒")
        else:
            hf_token = st.text_input("Hugging Face Token", type="password")
            if not hf_token:
                st.warning("Token needed for feedback.")
        
        st.markdown("---")
        st.info("**How it works:**\n1. **AI Grading:** BERT analyzes structure & vocab.\n2. **Feedback:** Qwen-72B gives teacher advice.")

    # 4. Main Content Layout
    st.markdown('<div class="main-header">🎓 AI Chinese Essay Grader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Instant Scoring & Personalized Feedback for Hong Kong Students</div>', unsafe_allow_html=True)

    # Input Area
    essay_text = st.text_area("📝 Paste Student Essay Here:", height=250, placeholder="在此輸入作文...")
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        grade_btn = st.button("✨ Grade Essay")

    # 5. Logic Execution
    if grade_btn:
        if not essay_text.strip():
            st.toast("⚠️ Please enter an essay first!", icon="🚫")
            return

        # Progress UI
        progress_text = st.empty()
        bar = st.progress(0)

        # Step A: Scoring
        scorer = load_scoring_pipeline()
        if scorer:
            progress_text.text("🧠 Analyzing text structure...")
            bar.progress(30)
            
            try:
                # Get raw probabilities
                predictions = scorer(essay_text, truncation=True, max_length=512, top_k=None)
                
                # Calculate Weighted Score
                score, level_text, theme_color = calculate_weighted_score(predictions)
                
            except Exception as e:
                st.error(f"Scoring Failed: {e}")
                bar.empty()
                return

            # Step B: Feedback
            progress_text.text("✍️ Teacher is writing comments...")
            bar.progress(70)
            
            feedback = generate_feedback(essay_text, score, level_text, hf_token)
            
            bar.progress(100)
            progress_text.empty()
            bar.empty()

            # Step C: Display Results
            st.markdown("---")
            c1, c2 = st.columns([1, 2], gap="large")

            with c1:
                st.markdown(f"""
                <div class="score-card" style="border-top: 6px solid {theme_color};">
                    <div style="color: #888; font-weight: bold; letter-spacing: 1px;">PROFICIENCY</div>
                    <div class="score-value">{score}</div>
                    <div class="score-label" style="background-color: {theme_color};">{level_text.split('(')[0]}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="feedback-box">
                    <h3 style="margin-top:0; color:#2C3E50;">👩‍🏫 Teacher's Feedback</h3>
                    <div style="color: #555; line-height: 1.6; font-size: 1.1rem;">
                        {feedback}
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
