# src/utils/record.py
import os
from omegaconf import DictConfig


def record_result(
        cfg: DictConfig,
        model_name: str,
        max_new_tokens: int,
        prompt: str,
        result_text: str,
        image_path: str,
        generation_kwargs: dict,
        generation_mode: str):

    folder_path = f"{cfg.save_dir.root + cfg.model_name}"
    if folder_path is None:
        raise ValueError(f"Unknown model name: {model_name}")

    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{model_name}_{max_new_tokens}.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("【Prompt】\n")
        f.write(prompt + "\n")
        f.write("【Result】\n")
        f.write(result_text + "\n")
        f.write("【Image Path】\n")
        f.write(image_path + "\n")
        f.write("【Generation Mode】\n")
        f.write(f"{generation_mode}\n")
        f.write("【Generation kwargs】\n")
        for key, value in generation_kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n\n")

    print(f"Result saved to: {file_path}")
