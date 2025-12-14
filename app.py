import streamlit as st
from transformers import pipeline
import torch
import gc

# --- Page Config ---
st.set_page_config(page_title="Dual Model Chat", page_icon="⚡", layout="wide")
st.title("⚡ Dual Model Interface: Fine-Tune vs Qwen")

# --- Model Paths ---
# UPDATE THIS with your actual Fine-Tuned Model path or Hugging Face ID
# If using Colab, this path must exist in the Colab instance (e.g., /content/drive/MyDrive/...)
PIPELINE_1_MODEL_ID = "/content/drive/MyDrive/my_finetuned_model" 
PIPELINE_2_MODEL_ID = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "Pipeline 1: Fine-Tuned Model"

# --- Sidebar ---
st.sidebar.header("⚙️ Control Panel")
selected_model = st.sidebar.radio(
    "Select Active Pipeline:",
    ("Pipeline 1: Fine-Tuned Model", "Pipeline 2: Qwen1.5-7B (Int4)")
)

# Clear chat if model changes to avoid context confusion
if selected_model != st.session_state.current_model:
    st.session_state.messages = []
    st.session_state.current_model = selected_model
    # Force garbage collection to free up VRAM for the new model
    gc.collect()
    torch.cuda.empty_cache()
    st.toast(f"Switched to {selected_model}. Memory cleared.", icon="🧹")

# --- Model Loading Functions (Cached) ---

@st.cache_resource
def get_pipeline_1():
    """Loads the User's Fine-Tuned Model"""
    print("Loading Pipeline 1...")
    return pipeline(
        "text-generation",
        model=PIPELINE_1_MODEL_ID,
        model_kwargs={"device_map": "auto"},
        torch_dtype=torch.float16
    )

@st.cache_resource
def get_pipeline_2():
    """Loads Qwen1.5-7B-Chat Int4"""
    print("Loading Pipeline 2 (Qwen)...")
    return pipeline(
        "text-generation",
        model=PIPELINE_2_MODEL_ID,
        model_kwargs={"device_map": "auto", "use_cache": True},
        torch_dtype=torch.float16
    )

# --- Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Type your message here..."):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner(f"Generating with {selected_model}..."):
            try:
                # Load ONLY the selected pipeline
                if selected_model == "Pipeline 1: Fine-Tuned Model":
                    pipe = get_pipeline_1()
                    # Pipeline 1 Params (Edit as needed)
                    outputs = pipe(
                        prompt, 
                        max_new_tokens=100, 
                        do_sample=True,
                        temperature=0.7
                    )
                    # Extract text for standard completion models
                    response_text = outputs[0]["generated_text"]
                    # If the model repeats the prompt, strip it out
                    if response_text.startswith(prompt):
                        response_text = response_text[len(prompt):]

                else:
                    # Pipeline 2 (Qwen)
                    pipe = get_pipeline_2()
                    # Qwen expects a list of messages
                    outputs = pipe(
                        st.session_state.messages,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95
                    )
                    response_text = outputs[0]["generated_text"][-1]["content"]

                # 3. Display and Save Response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Error: {e}")
                st.warning("If you see an 'OOM' (Out of Memory) error, restart the runtime. Running two models on one GPU is tight!")
