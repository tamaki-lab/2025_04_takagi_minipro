import os
import json
import csv
import glob
import hydra
from omegaconf import DictConfig
from typing import Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Paths
CSV_SAVE_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/qwen/batch_visualgenome_result/result.csv"
VG_IMAGE_DIR = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/images/VG_100K"
SCENE_GRAPHS_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/scene_graphs/scene_graphs.json"
REGION_DESC_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/region_descriptions/region_descriptions.json"

# Caches


def load_scene_graph() -> list:
    if not hasattr(load_scene_graph, "_cache"):
        with open(SCENE_GRAPHS_PATH, "r") as f:
            load_scene_graph._cache = json.load(f)
    return load_scene_graph._cache


def load_region_desc() -> list:
    if not hasattr(load_region_desc, "_cache"):
        with open(REGION_DESC_PATH, "r") as f:
            load_region_desc._cache = json.load(f)
    return load_region_desc._cache


def extract_vg_gt_info(image_id: int) -> dict:
    names = []
    # scene graphs
    for entry in load_scene_graph():
        if entry.get("image_id") == image_id:
            label = entry.get("regions", [{}])[0].get("phrase", "")
            seen = set()
            for obj in entry.get("objects", []):
                for name in obj.get("names", []):
                    if name not in seen:
                        names.append(name)
                        seen.add(name)
            return {"label": label or "unknown", "placeholders": names[:5]}
    # fallback region descriptions
    for entry in load_region_desc():
        if entry.get("image_id") == image_id and entry.get("regions"):
            label = entry["regions"][0].get("phrase", "")
            return {"label": label or "unknown", "placeholders": []}
    return {"label": "unknown", "placeholders": []}


def record_folder_result_csv_vg(
    csv_path, image_id, prediction, gt_json, generation_mode, generation_kwargs, is_first=False
):
    fieldnames = ["id", "result", "ground_truth_json", "generation_mode", "generation_kwargs"]
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    row = {
        "id": image_id,
        "result": prediction,
        "ground_truth_json": json.dumps(gt_json, ensure_ascii=False),
        "generation_mode": generation_mode if is_first else "",
        "generation_kwargs": json.dumps(generation_kwargs, ensure_ascii=False) if is_first else ""
    }
    rows.append(row)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV updated: {csv_path}")


@hydra.main(config_path="/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/config",
            config_name="video_qwen", version_base=None)
def main(cfg: DictConfig):
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Qwen tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map torch_dtype string to actual dtype
    dtype_str = cfg.get("torch_dtype", "bf16")
    if dtype_str in ["bf16", "bfloat16"]:
        torch_dtype = torch.bfloat16
    elif dtype_str in ["fp16", "float16"]:
        torch_dtype = torch.float16
    else:
        try:
            torch_dtype = getattr(torch, dtype_str)
        except AttributeError:
            raise ValueError(f"Unsupported torch_dtype: {dtype_str}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype
    ).eval()

    # Generation config
    gen_cfg = GenerationConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    model.generation_config = gen_cfg

    # Prepare image list
    image_paths = sorted(
        glob.glob(os.path.join(VG_IMAGE_DIR, "*.jpg")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )[: cfg.num_samples]

    for idx, image_path in enumerate(image_paths):
        print(f"Processing ({idx + 1}/{len(image_paths)}): {image_path}")
        # Build multimodal query
        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": cfg.question}
        ])
        # Generate response
        response, _ = model.chat(tokenizer, query=query, history=None)

        # Ground truth extraction
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])
        gt_info = extract_vg_gt_info(image_id)

        # Record CSV
        record_folder_result_csv_vg(
            CSV_SAVE_PATH,
            image_id,
            response,
            gt_info,
            gen_cfg.get_generation_mode(),
            dict(cfg),
            is_first=(idx == 0)
        )
    print("Qwen batch Visual Genome processing complete.")


if __name__ == "__main__":
    main()
