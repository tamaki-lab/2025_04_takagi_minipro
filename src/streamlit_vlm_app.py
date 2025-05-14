import shutil
import tempfile
from model.vlm_model import ModelExecutor
from model.generation_config import GenerationConfig
from omegaconf import OmegaConf
import streamlit as st
import os


st.title("VLM Model Executor (LLaMA / LLaVA / SmolVLM)")

# 入力: モデル設定
model_name = st.selectbox("Model", ["llama", "llava", "smolvlm"])
image_path = st.text_input(
    "Image Path",
    value="/mnt/NAS-TVS872XT/dataset/something-something-v2/frames/15/15_000060.jpg")
prompt = st.text_area("Prompt", value="""<image>
You are given 4 video frames arranged in a 2x2 layout in temporal order from top-left to bottom-right.

  Your task is to describe the dynamic action that happens across these frames. Focus on how things change over time.
  Pay particular attention to:
  - **How things change between frames** (e.g., movement, placement, interaction)
  - **The object being acted upon** (what is being touched, moved, or manipulated)

  Return your output in the following strict JSON format:

  {
    "label": "...",
    "template": "...",
    "placeholders": [...]
  }

  ⚠ Important:
  - Use your own sentence based solely on what is shown in the image.
  - "label" must be a natural English sentence describing the action.
  - "template" should replace the key objects/entities in the label with [1], [2].
  - "placeholders" must contain only those replaced elements, in order.
  - Do NOT repeat the full label in the placeholders.
  - Do NOT return anything except the JSON above.
""")

# Generation parameters
max_new_tokens = st.slider("max_new_tokens", 10, 2048, 500)
num_beams = st.slider("num_beams", 1, 10, 1)
num_beam_groups = st.slider("num_beam_groups", 1, 10, 1)
do_sample = st.checkbox("do_sample", value=True)
temperature = st.slider("temperature", 0.1, 1.5, 0.7)
top_k = st.slider("top_k", 1, 50, 10)
penalty_alpha = st.slider("penalty_alpha", 0.0, 1.0, 0.0)
diversity_penalty = st.slider("diversity_penalty", 0.0, 1.0, 0.0)

if st.button("Run Model"):
    # 一時的な Hydra Config を作成
    tmp_dir = tempfile.mkdtemp()
    tmp_yaml_path = os.path.join(tmp_dir, "tmp_config.yaml")

    config_dict = {
        "model_name": model_name,
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
        "output_format": "text"
    }

    with open(tmp_yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(config_dict)))

    # Hydraをバイパスして手動でDictConfig作成・実行
    cfg = OmegaConf.load(tmp_yaml_path)
    gen_cfg = GenerationConfig.from_cfg(cfg)
    generation_mode = gen_cfg.get_generation_mode()
    executor = ModelExecutor(cfg)
    executor()

    base_filename = os.path.basename(cfg.image_path).replace(".jpg", "")
    save_path = os.path.join(
        "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/image_22",
        f"{base_filename}_2x2.jpg")

    txt_result_path = f"/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/{generation_mode}/llama_{max_new_tokens}_{generation_mode}.txt"
    with open(txt_result_path, "r", encoding="utf-8") as f:
        all_lines = f.read().split("==================================================")
        last_result = all_lines[-2] if len(all_lines) > 1 else all_lines[-1]

    st.image(save_path, caption="2x2 Combined Image", use_column_width=True)
    st.text_area("Generated Result", last_result.strip(), height=300)

    st.success("Model execution completed.")
    shutil.rmtree(tmp_dir)
