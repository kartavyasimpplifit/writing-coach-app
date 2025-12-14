import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import re
import time

# --- Page Config ---
st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# ==========================================================
# 🔧 CONFIGURATION
# ==========================================================

# 1. SCORING MODEL (Custom)
# We use the legacy API URL because custom models often live there.
URL_SCORING = "https://api-inference.huggingface.co/models/MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"

# 2. FEEDBACK MODEL (Super Lite Qwen)
# We use the new Router URL for standard models.
URL_FEEDBACK = "https://router.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"

def query_huggingface(payload, url, token):
    """
    sends the text to the Hugging Face Server using your SECURE TOKEN.
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ==========================================================
# 🖥️ MAIN APP UI
# ==========================================================

# --- Sidebar ---
with st.sidebar:
    st.title("📝 Writing Coach")
    st.caption("Platform: Streamlit Cloud")
    st.divider()
    
    # 🔐 TOKEN MANAGEMENT
    # We check if the token is in Streamlit Secrets (Best Practice)
    if "HUGGINGFACE_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        st.success("✅ Secure Token Found")
    else:
        st.warning("⚠️ No Token in Secrets")
        hf_token = st.text_input("Enter Hugging Face Token", type="password")
        st.caption("Get token: huggingface.co/settings/tokens")

# --- Header ---
st.header("🖊️ AI Essay Grading Assistant")
st.caption("Powered by Qwen 2.5 (0.5B) & Custom Fine-Tuned Model")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Student Submission")
    essay_text = st.text_area("Paste Essay Here", height=300, value="今天天氣真好。我和爸爸媽媽去了公園...")
    
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False
        
    if st.button("🚀 Analyze Essay", type="primary"):
        st.session_state.analyze_clicked = True

# ==========================================================
# 🧠 ANALYSIS LOGIC
# ==========================================================

if st.session_state.analyze_clicked:
    if not essay_text:
        st.warning("Please write an essay first.")
    elif not hf_token:
        st.error("❌ You must provide a Hugging Face Token to run these models.")
    else:
        
        # --- 1. RUN SCORING (MirandaZhao) ---
        with st.spinner("Pipeline 1: Scoring (Waking up custom model)..."):
            score_payload = {
                "inputs": f"Score this essay (0-100):\n{essay_text}",
                "parameters": {"max_new_tokens": 50, "return_full_text": False}
            }
            score_res = query_huggingface(score_payload, URL_SCORING, hf_token)

        # --- 2. RUN FEEDBACK (Qwen 0.5B) ---
        with st.spinner("Pipeline 2: Generating Feedback..."):
            feedback_payload = {
                "inputs": f"<|im_start|>system\nYou are a helpful writing coach. Give 3 short, specific tips in Chinese.<|im_end|>\n<|im_start|>user\nReview this essay:\n{essay_text}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {"max_new_tokens": 300, "temperature": 0.7, "return_full_text": False}
            }
            feedback_res = query_huggingface(feedback_payload, URL_FEEDBACK, hf_token)

        # --- 3. PROCESS RESULTS ---
        
        # Check for "Model Loading" Error (Common on Free Tier)
        #  -> Diagram showing the model sleeping vs waking
        p1_error = isinstance(score_res, dict) and "error" in score_res
        p2_error = isinstance(feedback_res, dict) and "error" in feedback_res

        if p1_error:
            err_msg = score_res.get('error', '')
            if "loading" in str(err_msg).lower():
                st.warning("⏳ The Scoring Model is waking up from sleep. Please wait 30 seconds and click 'Analyze' again.")
            else:
                st.error(f"Scoring Failed: {err_msg}")
        
        if p2_error:
            st.error(f"Feedback Failed: {feedback_res.get('error')}")

        # If both succeeded:
        if not p1_error and not p2_error:
            try:
                # Parse Score
                raw_score = score_res[0]['generated_text'] if isinstance(score_res, list) else str(score_res)
                # Find the first number in the output
                nums = re.findall(r'\d+', raw_score)
                final_score = int(nums[0]) if nums else 0

                # Parse Feedback
                feedback_txt = feedback_res[0]['generated_text'] if isinstance(feedback_res, list) else str(feedback_res)

                # Display
                st.success("Analysis Complete!")
                
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("Score", f"{final_score}/100")
                with m2:
                    st.info(feedback_txt)
                
                # Chart
                if final_score > 0:
                    df = pd.DataFrame({'Metric': ['Score'], 'Value': [final_score]})
                    st.plotly_chart(px.bar(df, x='Metric', y='Value', range_y=[0,100]), use_container_width=True)

            except Exception as e:
                st.error(f"Error parsing results: {e}")
