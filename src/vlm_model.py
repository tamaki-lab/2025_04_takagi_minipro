import os
from model.llama import Llama
from model.llava import Llava
from model.smolvlm import SmolVLM
from PIL import Image
import hydra
from omegaconf import DictConfig
import torch

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"


def record_result(
        model_name: str,
        max_new_tokens: int,
        prompt: str,
        result_text: str,
        image_path: str,
        generation_kwargs: dict):
    folder_mapping = {
        "llama": "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llama",
        "llava": "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/llava",
        "smolvlm": "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/smolVLM",
    }
    folder_path = folder_mapping.get(model_name.lower())
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
        f.write("【Generation kwargs】\n")
        for key, value in generation_kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n\n")


def execution_llama(cfg: DictConfig):
    image = [Image.open(cfg.image_path)]
    prompt = cfg.prompt
    next_prompt = cfg.next_prompt
    model = Llama(max_new_tokens=cfg.max_new_tokens).to(DEVICE)
    model.prepare_model()

    # 1回目の実験
    response, history = model(prompt, image)
    print(response)
    record_result(cfg.model_name, cfg.max_new_tokens, prompt, response, cfg.image_path, model.generation_kwargs)

    # 2回目の実験（連続対話）
    # next_response, _ = model(next_prompt, image, history)
    # print(next_response)
    # record_result(cfg.model_name, cfg.max_new_tokens, next_prompt, next_response, cfg.image_path)


def execution_llava(cfg: DictConfig):
    image = [Image.open(cfg.image_path)]
    prompt = cfg.prompt
    model = Llava(max_new_tokens=cfg.max_new_tokens).to(DEVICE)
    model.prepare_model()

    output = model(prompt, image)
    # 必要に応じて "ASSISTANT:" 以降のテキストを抽出
    assistant_output = output.split('ASSISTANT: ')[-1]
    print(assistant_output)
    record_result(cfg.model_name, cfg.max_new_tokens, prompt, assistant_output, cfg.image_path, model.generation_kwargs)


def execution_smolvlm(cfg: DictConfig):
    image = Image.open(cfg.image_path)
    prompt = cfg.prompt
    print("Image loaded:", cfg.image_path)
    model = SmolVLM(max_new_tokens=cfg.max_new_tokens).to(DEVICE)
    model.prepare_model()

    generated_text = model(prompt, image)
    print(generated_text)
    record_result(cfg.model_name, cfg.max_new_tokens, prompt, generated_text, cfg.image_path, model.generation_kwargs)


@hydra.main(config_path="../config", config_name="smolvlm")
def main(cfg: DictConfig):
    if cfg.model_name == "llama":
        execution_llama(cfg)
    elif cfg.model_name == "llava":
        execution_llava(cfg)
    elif cfg.model_name == "smolvlm":
        execution_smolvlm(cfg)


if __name__ == "__main__":
    main()
