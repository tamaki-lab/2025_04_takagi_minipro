# image_qwen_simple.py

import os
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

torch.manual_seed(1234)


@hydra.main(config_path="/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/config",
            config_name="video_qwen", version_base=None)
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_device)
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )

    query = tokenizer.from_list_format([
        {"image": cfg.image_path},
        {"text": cfg.question}
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print("=== Response ===")
    print(response)

    out = cfg.result_txt
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"Image: {cfg.image_path}\n")
        f.write(f"Response: {response}\n")
        f.write("-----\n")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
