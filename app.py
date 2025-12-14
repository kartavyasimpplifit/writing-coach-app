import streamlit as st
import torch
import gc
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Essay Coach & Scorer", page_icon="📝", layout="wide")
st.title("📝 Essay Scoring & Coaching Assistant")

# --- Model Definitions ---
# Pipeline 1: Your Fine-Tuned Model (Hosted on Hugging Face)
MODEL_1_ID = "MirandaZhao/Finetuned_Essay_Scoring_Model_Epoch3"
# Pipeline 2: Chat Model (Quantized for Memory Efficiency)
MODEL_2_ID = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Selection")
model_option = st.sidebar.radio(
    "Select Active Function:",
    ("Pipeline 1: Essay Scoring (Fine-Tuned)", "Pipeline 2: Writing Coach (Qwen Chat)")
)

# --- Memory Management ---
def clear_gpu_memory():
    """Forces RAM release to prevent Colab crashes."""
    if "active_pipeline" in st.session_state and st.session_state.active_pipeline:
        del st.session_state.active_pipeline
    st.session_state.active_pipeline = None
    gc.collect()
    torch.cuda.empty_cache()

# --- Model Loading Logic ---
if "current_model_name" not in st.session_state:
    st.session_state.current_model_name = None

# If the user switches the model in the sidebar, we reset the GPU
if st.session_state.current_model_name != model_option:
    clear_gpu_memory()
    st.session_state.current_model_name = model_option
    st.toast(f"Switched to {model_option}. Memory cleared.", icon="🧹")

# --- Main App Logic ---

# 1. Load the selected model only when needed
if "active_pipeline" not in st.session_state or st.session_state.active_pipeline is None:
    with st.spinner(f"Loading {model_option}... (This takes ~1 min)"):
        try:
            if model_option == "Pipeline 1: Essay Scoring (Fine-Tuned)":
                # NOTE: If your model is BERT-based (Classification), change "text-generation" to "text-classification"
                # Assuming it is a generative model based on your previous requests:
                pipe = pipeline(
                    "text-generation",
                    model=MODEL_1_ID,
                    model_kwargs={"device_map": "auto"},
                    torch_dtype=torch.float16
                )
            else:
                # Load Qwen Int4
                pipe = pipeline(
                    "text-generation",
                    model=MODEL_2_ID,
                    model_kwargs={"device_map": "auto", "use_cache": True},
                    torch_dtype=torch.float16
                )
            st.session_state.active_pipeline = pipe
            st.success("Model Ready!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

# 2. Chat/Input Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear history on model switch to avoid context confusion
if "history_tracker" not in st.session_state:
    st.session_state.history_tracker = model_option
if st.session_state.history_tracker != model_option:
    st.session_state.messages = []
    st.session_state.history_tracker = model_option

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
input_label = "Paste your essay here..." if "Essay Scoring" in model_option else "Ask your writing coach..."
if prompt := st.chat_input(input_label):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                pipe = st.session_state.active_pipeline
                
                if "Qwen" in model_option:
                    # Qwen Chat Logic
                    outputs = pipe(
                        st.session_state.messages,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7
                    )
                    response = outputs[0]["generated_text"][-1]["content"]
                else:
                    # Scoring Model Logic
                    # If it's a generation model, it will generate text.
                    # If it's a classification model (BERT), this might need adjustment.
                    outputs = pipe(
                        prompt,
                        max_new_tokens=100,
                        do_sample=True
                    )
                    # Handle different output formats
                    if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                        raw_text = outputs[0]["generated_text"]
                        response = raw_text.replace(prompt, "").strip()
                    else:
                        response = str(outputs)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Processing Error: {e}")
                st.warning("If you see an OOM error, refresh the page to reset GPU memory.")
