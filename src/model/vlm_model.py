import os
from PIL import Image
import hydra
from omegaconf import DictConfig
from .model_factory import configure_model
from .generation_config import GenerationConfig


def record_result(
        cfg: DictConfig,
        model_name: str,
        max_new_tokens: int,
        prompt: str,
        result_text: str,
        image_path: str,
        generation_kwargs: dict):
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
        f.write("【Generation kwargs】\n")
        for key, value in generation_kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n\n")

    print(f"Result saved to: {file_path}")


class ModelExecutor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_info = GenerationConfig.from_cfg(cfg)
        self.model = configure_model(cfg, self.model_info)

    def __call__(self):
        # Load image and prompt
        image = [Image.open(self.cfg.image_path)]
        prompt = self.cfg.prompt

        # Prepare and execute the model
        self.model.prepare_model()
        output = self.model.forward(prompt, image)

        if isinstance(output, tuple):
            result_text = output[0]
        else:
            result_text = output

        # Record the result
        record_result(
            cfg=self.cfg,
            model_name=self.cfg.model_name,
            max_new_tokens=self.model_info.max_new_tokens,
            prompt=prompt,
            result_text=result_text,
            image_path=self.cfg.image_path,
            generation_kwargs=self.model.generation_kwargs
        )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    executor = ModelExecutor(cfg)
    executor()
    print("recorded result")


if __name__ == "__main__":
    main()
