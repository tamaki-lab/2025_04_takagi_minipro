# batch_vlm_runner.py (Hydra形式)
from model.generation_config import GenerationConfig
import os
import json
from omegaconf import DictConfig
from copy import deepcopy
import hydra
from model.record_result import (
    record_folder_result_csv,
    record_result,
)
from model.vlm_model import ModelExecutor
from model.image_utils import concat_images_2x2_from_base


def run_model_and_get_result(cfg: DictConfig) -> str:
    """
    ModelExecutorを使って推論を実行し、result_textのみを取得する。
    """
    executor = ModelExecutor(cfg)
    image, _ = concat_images_2x2_from_base(cfg.image_path)
    image = [image]

    save_dir = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/image_22"
    os.makedirs(save_dir, exist_ok=True)

    base_filename = os.path.basename(cfg.image_path).replace(".jpg", "")
    save_path = os.path.join(save_dir, f"{base_filename}_2x2.jpg")
    image[0].save(save_path)
    print(f"Combined 2x2 image saved to: {save_path}")

    prompt = cfg.prompt

    executor.model.prepare_model()
    output = executor.model.forward(prompt, image)

    result_text = output[0] if isinstance(output, tuple) else output
    # generation_mode = executor.model_info.get_generation_mode()

    # record_result(
    #     cfg=cfg,
    #     model_name=cfg.model_name,
    #     max_new_tokens=executor.model_info.max_new_tokens,
    #     prompt=prompt,
    #     result_text=result_text,
    #     image_path=cfg.image_path,
    #     generation_kwargs=executor.model.generation_kwargs,
    #     generation_mode=generation_mode,
    #     selected_images=selected_image
    # )

    return result_text


@hydra.main(config_path="../../config", config_name="llama", version_base=None)
def main(cfg: DictConfig):
    with open("/mnt/NAS-TVS872XT/dataset/something-something-v2/anno/something-something-v2-train.formatted.json", "r") as f:
        gt_data = json.load(f)
    gt_dict = {entry["id"]: entry for entry in gt_data}

    start_folder = cfg.get("start_folder", 1)
    num_samples = cfg.get("num_samples", 5)
    base_path = os.path.dirname(os.path.dirname(cfg.image_path))  # → /frames
    csv_path = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama/batch_SSv2_result/folder_summary1-100.csv"

    for i in range(start_folder, start_folder + num_samples):
        folder_path = os.path.join(base_path, str(i))
        if not os.path.exists(folder_path):
            print(f"Skipping folder {i} (does not exist)")
            continue

        jpgs = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        if not jpgs:
            print(f"Skipping folder {i} (no jpgs)")
            continue

        first_image_path = os.path.join(folder_path, jpgs[0])
        cfg_loop = deepcopy(cfg)
        cfg_loop.image_path = first_image_path

        print(f"Processing folder {i}: {first_image_path}")
        result = run_model_and_get_result(cfg_loop)

        gen_cfg = GenerationConfig.from_cfg(cfg_loop)
        generation_mode = gen_cfg.get_generation_mode()
        generation_kwargs = cfg_loop  # DictConfigそのまま渡せる
        gt_entry = gt_dict.get(str(i), {})

        record_folder_result_csv(csv_path, str(i), result,
                                 generation_mode=generation_mode,
                                 generation_kwargs=generation_kwargs,
                                 ground_truth=gt_entry)

    print("Batch VLM processing complete.")


if __name__ == "__main__":
    main()
