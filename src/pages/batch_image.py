import streamlit as st
import os
import sys
import tempfile
import shutil
import subprocess
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.title("Batch Folder Inference")

with st.form("batch_form"):
    start_folder = st.number_input("Start folder", min_value=1, value=1)
    image_path = st.text_input(
        "Image Path",
        value="/mnt/NAS-TVS872XT/dataset/something-something-v2/frames/15/15_000060.jpg")
    prompt = st.text_area("Prompt", value="""<image>
    This image contains 4 video frames arranged in a 2x2 grid in temporal order from top-left to bottom-right.

    Your task is to describe the main action taking place across the frames using the following **strict JSON format**.

    Example:
    {
    "label": "Putting book on shelf",
    "template": "Putting [something] on [something]",
    "placeholders": ["book", "shelf"]
    }
    """)

    # generation config
    max_new_tokens = st.slider("max_new_tokens", 10, 2048, 500)
    num_beams = st.slider("num_beams", 1, 10, 1)
    num_beam_groups = st.slider("num_beam_groups", 1, 10, 1)
    do_sample = st.checkbox("do_sample", value=True)
    temperature = st.slider("temperature", 0.1, 1.5, 0.7)
    top_k = st.slider("top_k", 1, 50, 10)
    penalty_alpha = st.slider("penalty_alpha", 0.0, 1.0, 0.0)
    diversity_penalty = st.slider("diversity_penalty", 0.0, 1.0, 0.0)
    start_folder = st.number_input("Start Folder", min_value=1, value=100)
    num_samples = st.number_input("Number of Samples", min_value=1, value=100)
    csv_filename = st.text_input("CSV filename (e.g. folder_summary1-100.csv)", value="folder_summary1-100.csv")

    run = st.form_submit_button("Run Batch")

if run:
    config = {
        "model_name": "llama",
        "image_path": image_path,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_beam_groups": num_beam_groups,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "penalty_alpha": penalty_alpha,
        "diversity_penalty": diversity_penalty,
        "use_pretrained": True,
        "torch_home": "/mnt/HDD10TB-148/takagi/pretrained_models",
        "save_dir": {
            "root": "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/"
        },
        "output_format": "text",
        "start_folder": start_folder,
        "num_samples": num_samples,
        "csv_filename": csv_filename
    }

    tmp_dir = tempfile.mkdtemp()
    yaml_path = os.path.join(tmp_dir, "tmp_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(config)))

    script = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/model/batch_vlm_runner.py"
    command = [
        "python3", script,
        f"hydra.run.dir={tmp_dir}",
        f"+config_path={tmp_dir}",
        f"+config_name=tmp_config"
    ]
    st.info("Running batch...")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        st.success("Batch completed")
    else:
        st.error("Batch failed")
        st.text(result.stderr)

    shutil.rmtree(tmp_dir)
