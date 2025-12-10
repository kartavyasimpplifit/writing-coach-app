import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipelines import ScoringPipeline, FeedbackPipeline

st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📝 Writing Coach")
    st.divider()

    # Check for Secret Key first
    if "HUGGINGFACE_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        st.success("API Token Loaded Securely! 🔒")
    else:
        hf_token = st.text_input("HuggingFace Token", type="password")

    st.info("Model: Qwen/Qwen2.5-7B-Instruct")

# --- Main App ---
st.header("🖊️ AI Essay Grading Assistant")

col1, col2 = st.columns([2, 1])

with col1:
    genre = st.selectbox("Genre (文體)", ["Narrative (記叙文)", "Argumentative (議論文)", "Expository (說明文)"])
    default_text = "今天天氣很好，我和家人去公園野餐。我們看到了很多漂亮的花朵，還吃了美味的三文治。這真是快樂的一天。"
    essay_text = st.text_area("Student Essay (學生作文)", height=300, value=default_text)

    analyze_btn = st.button("🚀 Analyze Essay", type="primary")

if analyze_btn:
    if not hf_token:
        st.error("Please add your Hugging Face Token in the sidebar or App Settings.")
    else:
        try:
            scoring_pipe = ScoringPipeline(hf_token)
            feedback_pipe = FeedbackPipeline(hf_token)

            with st.spinner("Step 1/2: Grading Essay..."):
                scores = scoring_pipe.run(essay_text, genre)

            if "error" in scores:
                st.error(f"Error: {scores['error']}")
            else:
                with st.spinner("Step 2/2: Generating Feedback..."):
                    feedback = feedback_pipe.run(essay_text, genre, scores)

                st.success("Grading Complete!")

                # 1. Score Metrics
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("Total Score", f"{scores.get('holistic_score', 0)} / 100")

                with m2:
                    dims = scores.get('dimensions', {})
                    df = pd.DataFrame(dict(r=list(dims.values()), theta=list(dims.keys())))
                    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0,100])
                    st.plotly_chart(fig, use_container_width=True)

                # 2. Feedback
                st.subheader("📝 Teacher's Comment")
                st.info(scores.get('brief_comment', "No comment generated."))

                st.subheader("🔍 Detailed Corrections")
                if "corrections" in feedback:
                    for item in feedback['corrections']:
                        with st.expander(f"Fix: {item.get('quote', 'Text')}"):
                            st.markdown(f"**Correction:** {item.get('fix')}")
                            st.caption(f"Reason: {item.get('reason')}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
