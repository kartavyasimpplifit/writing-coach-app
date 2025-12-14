import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
from github import Github
import json
import torch

# ==========================================
# PART 1: FRONTEND CONFIG & CSS
# ==========================================

st.set_page_config(
    page_title="AI Chinese Essay Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Awesome UI + Dark Text Fix)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');
    
    .main { background-color: #F8F9FA; font-family: 'Noto Sans TC', sans-serif; }
    
    /* FIX: Force Input Text Color to Dark Grey */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        caret-color: #333333;
        border-radius: 12px;
        border: 2px solid #E0E0E0;
        padding: 15px;
        font-size: 16px;
        font-family: "KaiTi", "SimKai", "Serif";
    }
    .stTextArea textarea:focus { border-color: #3498DB; box-shadow: 0 0 0 2px rgba(52,152,219,0.2); }
    
    /* Score Card */
    .score-card {
        background: white; border-radius: 15px; padding: 25px;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        height: 100%; display: flex; flex-direction: column;
        justify-content: center; align-items: center;
    }
    .score-value { font-size: 4.5rem; font-weight: 800; color: #2C3E50; margin: 10px 0; line-height: 1; }
    .score-label { font-size: 1.2rem; font-weight: 600; padding: 5px 15px; border-radius: 20px; color: white; }

    /* Feedback Box */
    .feedback-box {
        background: white; border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 5px solid #3498DB;
        height: 100%;
    }
    
    /* Header & Buttons */
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2C3E50; text-align: center; margin-top: -20px; }
    .sub-header { font-size: 1.1rem; color: #7F8C8D; text-align: center; margin-bottom: 2rem; }
    .stButton>button {
        width: 100%; border-radius: 30px;
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white; border: none; padding: 12px 0; font-weight: bold; font-size: 18px;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52,152,219,0.3); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# PART 2: BACKEND LOGIC
# ==========================================

@st.cache_resource
def load_scoring_pipeline():
    """Load BERT model locally (CPU optimized)."""
    model_id = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"⚠️ Scoring Model Error: {e}")
        return None

def calculate_weighted_score(predictions):
    """Calculate 0-100 score with multi-language support."""
    label_weights = {
        "不及格": 45, "Needs Improvement": 45, "LABEL_0": 45,
        "良好": 75, "Good": 75, "LABEL_1": 75,
        "優秀": 95, "Excellent": 95, "LABEL_2": 95,
        "优秀": 95 # Simplified Chinese support
    }

    weighted_score = 0
    total_confidence = 0
    
    for p in predictions:
        l = p['label']
        c = p['score']
        for key, val in label_weights.items():
            if key in l:
                weighted_score += (val * c)
                total_confidence += c
                break
    
    if total_confidence == 0: return 0, "Error", "#999"

    final_score = int(weighted_score)
    
    if final_score >= 85:
        return final_score, "Excellent (優秀)", "#2ECC71" # Green
    elif final_score >= 60:
        return final_score, "Good (良好)", "#F39C12" # Orange
    else:
        return final_score, "Needs Improvement (需改進)", "#E74C3C" # Red

def generate_feedback(essay, score, level, hf_token):
    """Generate feedback using Qwen-72B via API."""
    if not hf_token: return "⚠️ Token missing. Please add your Hugging Face token."
    
    client = InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
    prompt = f"""
    Role: Experienced Chinese teacher.
    Task: Grade this Primary Student essay.
    Essay: "{essay}"
    Score: {score}/100 ({level})
    Instructions:
    1. Tone: Encouraging.
    2. Format: 🌟 Highlights (1 sentence), 💡 Suggestion (1 tip for 'Show Don't Tell'), ✍️ Correction (fix 1 vocab).
    3. Language: Traditional Chinese.
    4. Length: Under 150 words.
    """
    try:
        msg = [{"role": "user", "content": prompt}]
        res = client.chat_completion(msg, max_tokens=500, temperature=0.7)
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ Feedback Error: {str(e)}"

def save_to_github(essay, score, level, correction):
    """Saves teacher feedback to a JSON file in YOUR GitHub Repo."""
    
    # --- CONFIGURATION: CHANGE THIS! ---
    REPO_NAME = "your-github-username/your-repo-name" 
    # Example: "MirandaZhao/essay-grader-app"
    
    if 'GITHUB_TOKEN' not in st.secrets:
        st.error("Missing GITHUB_TOKEN in Secrets.")
        return False

    g = Github(st.secrets["GITHUB_TOKEN"])
    try:
        repo = g.get_repo(REPO_NAME)
        file_path = "data/feedback.json"
        
        try:
            contents = repo.get_contents(file_path)
            existing_data = json.loads(contents.decoded_content.decode())
        except:
            existing_data = [] # Create new if missing
            
        new_entry = {
            "essay": essay,
            "ai_score": score,
            "ai_level": level,
            "teacher_correction": correction
        }
        existing_data.append(new_entry)
        
        updated_content = json.dumps(existing_data, indent=2, ensure_ascii=False)
        
        if 'contents' in locals():
            repo.update_file(file_path, "Add feedback", updated_content, contents.sha)
        else:
            repo.create_file(file_path, "Init database", updated_content)
            
        return True
    except Exception as e:
        st.error(f"GitHub Error: {e}")
        return False

# ==========================================
# PART 3: MAIN APP
# ==========================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3429/3429414.png", width=80)
        st.markdown("## 👩‍🏫 Teacher's Desk")
        if 'HF_TOKEN' in st.secrets:
            hf_token = st.secrets['HF_TOKEN']
            st.success("AI Token Loaded 🔒")
        else:
            hf_token = st.text_input("Enter HF Token", type="password")
        
        st.info("**System Status:**\n🟢 Scoring (Local)\n🟢 Feedback (Remote)\n🟢 Database (GitHub)")

    # Main Area
    st.markdown('<div class="main-header">🎓 AI Chinese Essay Grader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Instant Scoring & Personalized Feedback</div>', unsafe_allow_html=True)

    essay_text = st.text_area("📝 Paste Student Essay Here:", height=250, placeholder="在此輸入作文...")
    
    # Buttons
    col_grade, _ = st.columns([1, 3])
    with col_grade:
        grade_btn = st.button("✨ Grade Essay")

    if 'result_generated' not in st.session_state:
        st.session_state.result_generated = False

    if grade_btn:
        if not essay_text.strip():
            st.toast("⚠️ Input empty!", icon="🚫")
            return

        progress_text = st.empty()
        bar = st.progress(0)

        # 1. Scoring
        scorer = load_scoring_pipeline()
        if scorer:
            progress_text.text("🧠 Analyzing structure...")
            bar.progress(30)
            try:
                preds = scorer(essay_text, truncation=True, max_length=512, top_k=None)
                score, level_text, color = calculate_weighted_score(preds)
                
                # Save to session state so it persists for the feedback button
                st.session_state.score = score
                st.session_state.level_text = level_text
                st.session_state.color = color
                st.session_state.essay = essay_text
                
            except Exception as e:
                st.error(f"Scoring Failed: {e}")
                return

            # 2. Feedback
            progress_text.text("✍️ Drafting comments...")
            bar.progress(70)
            feedback = generate_feedback(essay_text, score, level_text, hf_token)
            st.session_state.feedback = feedback
            
            bar.progress(100)
            progress_text.empty()
            bar.empty()
            st.session_state.result_generated = True

    # Display Results (Persist after button click)
    if st.session_state.result_generated:
        st.markdown("---")
        c1, c2 = st.columns([1, 2], gap="large")

        with c1:
            st.markdown(f"""
            <div class="score-card" style="border-top: 6px solid {st.session_state.color};">
                <div style="color: #888; font-weight: bold;">PROFICIENCY</div>
                <div class="score-value">{st.session_state.score}</div>
                <div class="score-label" style="background-color: {st.session_state.color};">{st.session_state.level_text.split('(')[0]}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="feedback-box">
                <h3 style="margin-top:0; color:#2C3E50;">👩‍🏫 Teacher's Feedback</h3>
                <div style="color: #555; line-height: 1.6; font-size: 1.1rem;">
                    {st.session_state.feedback}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- TEACHER VALIDATION FORM ---
        st.markdown("---")
        st.subheader("🧐 Help Improve the AI")
        
        with st.form("feedback_form"):
            st.write(f"The AI graded this as: **{st.session_state.level_text}**")
            correction = st.radio(
                "Is this grade correct?",
                ["✅ Yes, correct", "⚠️ No, should be 'Needs Improvement'", "⚠️ No, should be 'Good'", "⚠️ No, should be 'Excellent'"],
                horizontal=True
            )
            submit_feed = st.form_submit_button("Submit Feedback to Database")
            
            if submit_feed:
                final_label = correction
                if "Yes" in correction:
                    final_label = st.session_state.level_text.split('(')[0] # Keep original
                else:
                    final_label = correction.replace("⚠️ No, should be ", "").replace("'", "")
                
                if save_to_github(st.session_state.essay, st.session_state.score, st.session_state.level_text, final_label):
                    st.success("✅ Feedback saved to GitHub! The model will learn from this next week.")
                else:
                    st.error("❌ Failed to save. Check your GITHUB_TOKEN and Repo Name.")

if __name__ == "__main__":
    main()
