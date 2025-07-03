import os
import json
import csv
import hydra
from omegaconf import DictConfig
from copy import deepcopy
from collections import Counter
from typing import Optional
from PIL import Image

from model.vlm_model import ModelExecutor
from model.generation_config import GenerationConfig

# パス設定
CSV_SAVE_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/batch_visualgenome_result/result.csv"
VG_IMAGE_DIR = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/images/VG_100K"
SCENE_GRAPHS_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/scene_graphs/scene_graphs.json"
REGION_DESC_PATH = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/visual_genome_v1.2/region_descriptions/region_descriptions.json"

# キャッシュ
_SCENE_GRAPH_CACHE: Optional[list] = None
_REGION_DESC_CACHE: Optional[list] = None


def load_scene_graph() -> list:
    global _SCENE_GRAPH_CACHE
    if _SCENE_GRAPH_CACHE is None:
        with open(SCENE_GRAPHS_PATH, "r") as f:
            _SCENE_GRAPH_CACHE = json.load(f)
    return _SCENE_GRAPH_CACHE


def load_region_desc() -> list:
    global _REGION_DESC_CACHE
    if _REGION_DESC_CACHE is None:
        with open(REGION_DESC_PATH, "r") as f:
            _REGION_DESC_CACHE = json.load(f)
    return _REGION_DESC_CACHE


def extract_vg_gt_info(image_id: int) -> dict:
    names = []

    # scene graphs 優先
    for entry in load_scene_graph():
        if entry.get("image_id") == image_id:
            label = ""
            if entry.get("regions"):
                label = entry["regions"][0].get("phrase", "")
            seen = set()
            for obj in entry.get("objects", []):
                if obj.get("names"):
                    name = obj["names"][0]
                    if name not in seen:
                        names.append(name)
                        seen.add(name)
            return {
                "label": label or "unknown",
                "placeholders": names[:5]
            }

    # fallback: region_descriptions
    for entry in load_region_desc():
        if entry.get("image_id") == image_id and entry.get("regions"):
            label = entry["regions"][0].get("phrase", "")
            return {
                "label": label or "unknown",
                "placeholders": names[:5]
            }

    return {
        "label": "unknown",
        "placeholders": []
    }


def record_folder_result_csv_vg(
        csv_path,
        image_id,
        prediction,
        gt_json,
        generation_mode,
        generation_kwargs,
        is_first=False):
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
        "generation_kwargs": "\n".join(f"{k}: {v}" for k, v in generation_kwargs.items()) if is_first else ""
    }

    rows.append(row)

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV updated: {csv_path}")


def run_model_and_get_result(cfg: DictConfig) -> str:
    executor = ModelExecutor(cfg)
    image = [Image.open(cfg.image_path)]
    prompt = cfg.prompt
    executor.model.prepare_model()
    output = executor.model.forward(prompt, image)
    return output[0] if isinstance(output, tuple) else output


@hydra.main(config_path="../../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    all_images = sorted(
        [os.path.join(VG_IMAGE_DIR, f) for f in os.listdir(VG_IMAGE_DIR) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

    for idx, image_path in enumerate(all_images[:cfg.num_samples]):
        cfg_loop = deepcopy(cfg)
        cfg_loop.image_path = image_path
        print(f"Processing: {image_path}")

        try:
            result = run_model_and_get_result(cfg_loop)
        except Exception as e:
            print(f"Skipping {image_path} due to error: {e}")
            continue

        image_id_str = os.path.basename(image_path)
        image_id_num = int(os.path.splitext(image_id_str)[0])
        gt_info = extract_vg_gt_info(image_id=image_id_num)

        gen_cfg = GenerationConfig.from_cfg(cfg_loop)
        generation_mode = gen_cfg.get_generation_mode()

        record_folder_result_csv_vg(
            CSV_SAVE_PATH,
            image_id=image_id_str,
            prediction=result,
            gt_json=gt_info,
            generation_mode=generation_mode,
            generation_kwargs=dict(cfg_loop),
            is_first=(idx == 0)
        )

    print("Visual Genome batch processing complete.")


if __name__ == "__main__":
    main()
