import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import time
import re

# --- Page Config ---
st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# --- API CONFIGURATION (FIXED) ---

# 🔴 Pipeline 1: Custom User Models MUST use the 'api-inference' URL
API_URL_SCORING = "https://api-inference.huggingface.co/models/MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"

# 🟢 Pipeline 2: Popular Models (Qwen) work better on the 'router' URL
API_URL_FEEDBACK = "https://router.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

def query_model(payload, api_url, token):
    """Sends request to HF. Handles non-JSON errors gracefully."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        # 1. Handle HTTP Errors (404, 500, 503)
        if response.status_code != 200:
            return {"error": f"Status {response.status_code}: {response.text}"}
            
        # 2. Try to decode JSON
        return response.json()
        
    except json.JSONDecodeError:
        # This catches the "line 1 column 1" error
        return {"error": f"Invalid API Response (Not JSON). Raw output: {response.text[:200]}..."}
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
    
    tab_text, tab_debug = st.tabs(["⌨️ Essay Content", "🛠️ Debug"])
    
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
        st.error("❌ Please provide a Hugging Face API Token.")
    else:
        # --- PIPELINE 1: SCORING ---
        with st.spinner("Pipeline 1: Scoring (This might take 30s)..."):
            # Note: We ask for 'return_full_text': False to just get new tokens
            scoring_payload = {
                "inputs": f"Score this narrative essay (0-100):\n{essay_text}",
                "parameters": {"max_new_tokens": 50, "temperature": 0.1, "return_full_text": False}
            }
            raw_score_response = query_model(scoring_payload, API_URL_SCORING, hf_token)

        # --- PIPELINE 2: FEEDBACK ---
        with st.spinner("Pipeline 2: Generating Feedback..."):
            feedback_payload = {
                "inputs": f"<|im_start|>system\nYou are a writing coach.<|im_end|>\n<|im_start|>user\nReview this essay. Provide 3 tips.\n\nEssay:\n{essay_text}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
            }
            raw_feedback_response = query_model(feedback_payload, API_URL_FEEDBACK, hf_token)

        # --- PROCESS & DISPLAY ---
        
        # 1. Check for Errors
        p1_error = isinstance(raw_score_response, dict) and "error" in raw_score_response
        p2_error = isinstance(raw_feedback_response, dict) and "error" in raw_feedback_response

        if p1_error:
            st.error(f"Pipeline 1 Failed: {raw_score_response['error']}")
            # Fallback if model is loading (Common on Free Tier)
            if "loading" in str(raw_score_response.get('error', '')).lower():
                st.warning("⏳ The Scoring Model is waking up. Please click 'Analyze' again in 30 seconds.")
        
        if p2_error:
            st.error(f"Pipeline 2 Failed: {raw_feedback_response['error']}")

        if not p1_error and not p2_error:
            # 2. Parse Data
            try:
                # Get text from list or dict
                score_out = raw_score_response[0]['generated_text'] if isinstance(raw_score_response, list) else str(raw_score_response)
                feedback_out = raw_feedback_response[0]['generated_text'] if isinstance(raw_feedback_response, list) else str(raw_feedback_response)

                # Extract Number
                numbers = re.findall(r'\d+', score_out)
                final_score = int(numbers[0]) if numbers else 0
                
                # Show Results
                st.success("Analysis Complete")
                
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("Score", f"{final_score}/100")
                with m2:
                    st.subheader("Coach Feedback")
                    st.write(feedback_out)
                
                if final_score > 0:
                    df = pd.DataFrame({'Metric': ['Score', 'Content', 'Grammar'], 'Value': [final_score, final_score, final_score]})
                    st.plotly_chart(px.bar(df, x='Metric', y='Value', range_y=[0,100]), use_container_width=True)

            except Exception as e:
                st.error(f"Parsing Error: {e}")

        # Debug Tab
        with tab_debug:
            st.write("Scoring Raw:", raw_score_response)
            st.write("Feedback Raw:", raw_feedback_response)
