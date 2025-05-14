from model.vlm_model import ModelExecutor
import streamlit as st
import os
import sys
import tempfile
import shutil
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.title("Single Image Inference")

# フォーム入力
with st.form("single_form"):
    model_name = st.selectbox("Model", ["llama", "llava", "smolvlm"])
    image_path = st.text_input("Image Path", "/mnt/NAS-TVS872XT/dataset/something-something-v2/frames/15/15_000060.jpg")
    prompt = st.text_area("Prompt", "<image> Describe this...")

    # generation config
    max_new_tokens = st.slider("max_new_tokens", 10, 2048, 500)
    num_beams = st.slider("num_beams", 1, 10, 1)
    num_beam_groups = st.slider("num_beam_groups", 1, 10, 1)
    do_sample = st.checkbox("do_sample", value=True)
    temperature = st.slider("temperature", 0.1, 1.5, 0.7)
    top_k = st.slider("top_k", 1, 50, 10)
    penalty_alpha = st.slider("penalty_alpha", 0.0, 1.0, 0.0)
    diversity_penalty = st.slider("diversity_penalty", 0.0, 1.0, 0.0)

    submitted = st.form_submit_button("Run")

if submitted:
    config = {
        "model_name": model_name,
        "image_path": image_path,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_beam_groups": 1,
        "do_sample": do_sample,
        "temperature": 0.7,
        "top_k": top_k,
        "penalty_alpha": penalty_alpha,
        "diversity_penalty": diversity_penalty,
        "use_pretrained": True,
        "torch_home": "/mnt/HDD10TB-148/takagi/pretrained_models",
        "save_dir": {"root": "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/"},
        "output_format": "text"
    }

    cfg = OmegaConf.create(config)
    executor = ModelExecutor(cfg)
    executor()

    st.success("Execution completed.")
