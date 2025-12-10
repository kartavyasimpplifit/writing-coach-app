import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipelines import ScoringPipeline, FeedbackPipeline
from PIL import Image
import time

st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📝 Writing Coach")
    st.caption("AI Essay Grading Demo")
    st.divider()
    
    # --- AUTOMATIC TOKEN DETECTION ---
    # 1. Try to get token from Secrets (Cloud or Local)
    if "HUGGINGFACE_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        st.success("✅ System Online (Token Loaded)")
    # 2. If no secret found, ask user manually
    else:
        st.warning("⚠️ No Token Found in Secrets")
        hf_token = st.text_input("Enter HuggingFace Token", type="password")
    
    st.info("Model: Qwen/Qwen2.5-7B-Instruct")
    st.markdown("---")
    st.markdown("**Demo Guide:**\n1. Select a Genre.\n2. Choose 'Type Text' or 'Upload Image'.\n3. Click Analyze.")

# --- Main App ---
st.header("🖊️ AI Essay Grading Assistant")
st.caption("Designed for Hong Kong Secondary Schools (Output translated for Demo)")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Student Submission")
    genre = st.selectbox(
        "Select Genre (文體)", 
        ["Narrative (記叙文)", "Argumentative (議論文)", "Expository (說明文)"]
    )
    
    # Input Tabs
    tab_text, tab_image = st.tabs(["⌨️ Type / Paste Text", "📷 Upload Handwriting"])
    
    essay_text = ""
    analyze_btn = False

    with tab_text:
        default_text = """今天天氣真好。我和爸爸媽媽去了公園。公園裡有很多人，有的在跑步，有的在放風箏。我看到了一朵紅色的花，很漂亮。我還吃了一個冰淇淋，是巧克力味的。
        
但是，回家的路上，我看到一個人亂扔垃圾。我覺得這是不對的。我們應該愛護環境。如果每個人都亂扔垃圾，地球就會變成垃圾場。雖然我只是一個小學生，但我也要保護地球。"""
        
        text_input = st.text_area(
            "Essay Content (Chinese)", 
            height=300, 
            value=default_text,
            key="text_area"
        )
        if st.button("🚀 Analyze Text", type="primary", key="btn_text"):
            essay_text = text_input
            analyze_btn = True

    with tab_image:
        st.info("Feature: Optical Character Recognition (OCR) for Handwritten Chinese")
        uploaded_file = st.file_uploader("Upload an image of the essay", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Student Handwriting', use_container_width=True)
            
            if st.button("🔍 Scan & Analyze Image", type="primary"):
                with st.spinner("Scanning handwriting (Simulated)..."):
                    time.sleep(2) 
                    essay_text = default_text 
                    analyze_btn = True
                    st.success("Text extracted successfully!")
                    with st.expander("View Extracted Text"):
                        st.write(essay_text)

# --- Analysis Logic ---
if analyze_btn:
    if not hf_token:
        st.error("❌ Error: API Token is missing. Please add it to Streamlit Secrets.")
    else:
        try:
            scoring_pipe = ScoringPipeline(hf_token)
            feedback_pipe = FeedbackPipeline(hf_token)
            
            progress_text = "AI is reading the essay..."
            my_bar = st.progress(0, text=progress_text)

            # Step 1: Scoring
            my_bar.progress(30, text="Analyzing rubric dimensions...")
            scores = scoring_pipe.run(essay_text, genre)
                
            if "error" in scores:
                st.error(f"Error: {scores['error']}")
            else:
                # Step 2: Feedback
                my_bar.progress(70, text="Generating English feedback...")
                feedback = feedback_pipe.run(essay_text, genre, scores)
                
                my_bar.progress(100, text="Complete!")
                my_bar.empty()
                
                st.success("✅ Analysis Complete!")
                
                # --- Result Area ---
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("Total Score", f"{scores.get('holistic_score', 0)} / 100")
                    st.markdown("### AI Verdict")
                    st.info(scores.get('brief_comment', "No comment generated."))
                
                with m2:
                    st.markdown("### Dimension Breakdown")
                    dims = scores.get('dimensions', {})
                    df = pd.DataFrame(dict(
                        Score=list(dims.values()), 
                        Dimension=list(dims.keys())
                    ))
                    fig = px.line_polar(df, r='Score', theta='Dimension', line_close=True, range_r=[0,100])
                    fig.update_traces(fill='toself')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🔍 Specific Corrections")
                    if "corrections" in feedback:
                        for item in feedback['corrections']:
                            with st.expander(f"Issue: ...{item.get('quote', '')[:10]}..."):
                                st.markdown(f"**Fix:** `{item.get('fix')}`")
                                st.markdown(f"**Reason:** *{item.get('reason')}*")
                
                with c2:
                    st.subheader("🚀 Strategic Advice")
                    if "suggestions" in feedback:
                        for i, s in enumerate(feedback['suggestions']):
                            st.info(f"**Tip {i+1}:** {s}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
