import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipelines import ScoringPipeline, FeedbackPipeline

st.set_page_config(page_title="HK Writing Coach AI", page_icon="📝", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📝 Writing Coach")
    st.caption("AI Essay Grading Demo")
    st.divider()
    
    # Check for Secret Key first
    if "HUGGINGFACE_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        st.success("API Token Loaded! 🔒")
    else:
        hf_token = st.text_input("HuggingFace Token", type="password")
    
    st.info("Model: Qwen/Qwen2.5-7B-Instruct")
    st.markdown("---")
    st.markdown("**Demo Guide:**\n1. Select a Genre.\n2. The sample text is a typical Grade 8 Chinese essay.\n3. Click Analyze to see the AI grade it in real-time.")

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
    
    # A slightly longer, more realistic sample essay for the demo
    default_text = """今天天氣真好。我和爸爸媽媽去了公園。公園裡有很多人，有的在跑步，有的在放風箏。我看到了一朵紅色的花，很漂亮。我還吃了一個冰淇淋，是巧克力味的。
    
但是，回家的路上，我看到一個人亂扔垃圾。我覺得這是不對的。我們應該愛護環境。如果每個人都亂扔垃圾，地球就會變成垃圾場。雖然我只是一個小學生，但我也要保護地球。"""
    
    essay_text = st.text_area(
        "Essay Content (Chinese)", 
        height=300, 
        value=default_text,
        help="Paste the Chinese essay text here."
    )
    
    analyze_btn = st.button("🚀 Analyze Essay", type="primary")

if analyze_btn:
    if not hf_token:
        st.error("⚠️ Token missing. Please check your settings.")
    else:
        try:
            scoring_pipe = ScoringPipeline(hf_token)
            feedback_pipe = FeedbackPipeline(hf_token)
            
            # Progress Bar for Visual Effect
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
                
                # 1. Score Metrics
                m1, m2 = st.columns([1, 2])
                with m1:
                    st.metric("Total Score", f"{scores.get('holistic_score', 0)} / 100")
                    st.markdown("### AI Verdict")
                    st.info(scores.get('brief_comment', "No comment generated."))
                
                with m2:
                    st.markdown("### Dimension Breakdown")
                    dims = scores.get('dimensions', {})
                    # Create Radar Chart
                    df = pd.DataFrame(dict(
                        Score=list(dims.values()), 
                        Dimension=list(dims.keys())
                    ))
                    fig = px.line_polar(df, r='Score', theta='Dimension', line_close=True, range_r=[0,100])
                    fig.update_traces(fill='toself')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()

                # 2. Detailed Feedback
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🔍 Specific Corrections")
                    st.caption("The AI found these issues in the Chinese text:")
                    if "corrections" in feedback:
                        for item in feedback['corrections']:
                            with st.expander(f"Issue in: ...{item.get('quote', '')[:10]}..."):
                                st.markdown(f"**Original:** `{item.get('quote')}`")
                                st.markdown(f"**Suggested Fix:** `{item.get('fix')}`")
                                st.markdown(f"**Reason (English):** *{item.get('reason')}*")
                    else:
                        st.write("No specific errors found.")
                
                with c2:
                    st.subheader("🚀 Strategic Advice")
                    st.caption("Actionable steps for the student:")
                    if "suggestions" in feedback:
                        for i, s in enumerate(feedback['suggestions']):
                            st.info(f"**Tip {i+1}:** {s}")
                    else:
                        st.write("No suggestions available.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
