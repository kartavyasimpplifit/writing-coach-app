import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import time

# --- Page Config ---
st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# --- API CONFIGURATION (UPDATED) ---
# 🔴 OLD: https://api-inference.huggingface.co/models/...
# 🟢 NEW: https://router.huggingface.co/models/...

# Pipeline 1: Your Fine-Tuned Model
API_URL_SCORING = "https://router.huggingface.co/models/MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"

# Pipeline 2: Qwen 2.5 7B (Instruct)
API_URL_FEEDBACK = "https://router.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

def query_model(payload, api_url, token):
    """Sends the text to Hugging Face and returns the JSON response."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

# --- Sidebar ---
with st.sidebar:
    st.title("📝 Writing Coach")
    st.caption("Mode: Narrative (記叙文)")
    st.divider()
    
    # Token Management
    if "HUGGINGFACE_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        st.success("✅ System Online")
    else:
        st.warning("⚠️ Token Missing")
        hf_token = st.text_input("Enter HuggingFace Token", type="password")
        st.caption("Required to access the models.")

# --- Main App ---
st.header("🖊️ AI Essay Grading Assistant")
st.caption("Pipeline 1: Fine-Tuned Model | Pipeline 2: Qwen 2.5 Instruct")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Student Submission")
    st.info("Genre Locked: Narrative (記叙文)")
    
    tab_text, tab_debug = st.tabs(["⌨️ Essay Content", "🛠️ Debug Raw Output"])
    
    essay_text = ""
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    with tab_text:
        default_text = "今天天氣真好。我和爸爸媽媽去了公園。公園裡有很多人..."
        text_input = st.text_area("Write your narrative essay here:", value=default_text, height=300)
        
        if st.button("🚀 Analyze Real Data", type="primary"):
            essay_text = text_input
            st.session_state.analyze_clicked = True

# --- Analysis Logic ---
if st.session_state.analyze_clicked:
    if not essay_text: essay_text = text_input
    
    if not hf_token:
        st.error("❌ Please provide a Hugging Face API Token to run the models.")
    else:
        # --- PIPELINE 1: SCORING ---
        with st.spinner("Pipeline 1: Running Fine-Tuned Scoring Model..."):
            # Payload for scoring
            scoring_payload = {
                "inputs": f"Score this narrative essay (0-100):\n{essay_text}",
                "parameters": {"max_new_tokens": 50, "temperature": 0.1, "return_full_text": False}
            }
            raw_score_response = query_model(scoring_payload, API_URL_SCORING, hf_token)

        # --- PIPELINE 2: FEEDBACK ---
        with st.spinner("Pipeline 2: Qwen is analyzing for feedback..."):
            # Payload for feedback
            feedback_payload = {
                "inputs": f"<|im_start|>system\nYou are a strict writing coach.<|im_end|>\n<|im_start|>user\nReview this narrative essay. Provide 3 specific improvements.\n\nEssay:\n{essay_text}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
            }
            raw_feedback_response = query_model(feedback_payload, API_URL_FEEDBACK, hf_token)

        # --- PROCESS & DISPLAY REAL DATA ---
        
        # 1. Check for API Errors (Model Loading / Auth Error)
        if isinstance(raw_score_response, dict) and "error" in raw_score_response:
            st.error(f"Pipeline 1 Error: {raw_score_response['error']}")
            st.warning("Note: Custom models on Hugging Face 'sleep' when not used. Wait 30s and try again.")
        elif isinstance(raw_feedback_response, dict) and "error" in raw_feedback_response:
            st.error(f"Pipeline 2 Error: {raw_feedback_response['error']}")
        else:
            # 2. Parse Real Data
            try:
                # Parsing Pipeline 1 (Scoring)
                score_text = raw_score_response[0]['generated_text'] if isinstance(raw_score_response, list) else str(raw_score_response)
                
                # Attempt to extract a number from the text
                import re
                numbers = re.findall(r'\d+', score_text)
                final_score = int(numbers[0]) if numbers else 0
                
                # Parsing Pipeline 2 (Feedback)
                feedback_text = raw_feedback_response[0]['generated_text'] if isinstance(raw_feedback_response, list) else str(raw_feedback_response)

                # Display Results
                st.success("✅ Real Analysis Complete")
                
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("AI Score", f"{final_score} / 100")
                    st.caption("Score extracted from Fine-Tuned Model")
                
                with m2:
                    st.subheader("Coach Feedback (Qwen)")
                    st.write(feedback_text)
                    
                # Visualization
                if final_score > 0:
                    df = pd.DataFrame({
                        'Metric': ['Holistic Score', 'Language Flow', 'Narrative Structure'],
                        'Value': [final_score, final_score, final_score] 
                    })
                    fig = px.bar(df, x='Metric', y='Value', range_y=[0,100])
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error parsing model output: {e}")
                st.warning("Check the 'Debug Raw Output' tab to see what the model actually returned.")

        # --- Debug Tab (For Transparency) ---
        with tab_debug:
            st.write("### Pipeline 1 Raw Response (Scoring)")
            st.json(raw_score_response)
            st.write("### Pipeline 2 Raw Response (Feedback)")
            st.json(raw_feedback_response)
