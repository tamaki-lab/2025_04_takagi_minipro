from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from .model_factory import configure_model
from .generation_config import GenerationConfig
from .record_result import record_result
from .image_utils import concat_images_2x2_from_base


class ModelExecutor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_info = GenerationConfig.from_cfg(cfg)
        self.model = configure_model(cfg, self.model_info)

    def __call__(self):
        # Load image and prompt
        # image = [Image.open(self.cfg.image_path)]
        image, selected_image = concat_images_2x2_from_base(self.cfg.image_path)
        image = [image]

        save_dir = "/mnt/HDD10TB-148/takagi/2025_04_takagi_minipro/src/result/image_22"
        os.makedirs(save_dir, exist_ok=True)

        # 保存ファイル名（例：from image path name or timestamp）
        base_filename = os.path.basename(self.cfg.image_path).replace(".jpg", "")
        save_path = os.path.join(save_dir, f"{base_filename}_2x2.jpg")
        image[0].save(save_path)

        print(f"Combined 2x2 image saved to: {save_path}")

        prompt = self.cfg.prompt

        # Prepare and execute the model
        self.model.prepare_model()
        output = self.model.forward(prompt, image)

        if isinstance(output, tuple):
            result_text = output[0]
        else:
            result_text = output

        generation_mode = self.model_info.get_generation_mode()

        # Record the result
        record_result(
            cfg=self.cfg,
            model_name=self.cfg.model_name,
            max_new_tokens=self.model_info.max_new_tokens,
            prompt=prompt,
            result_text=result_text,
            image_path=self.cfg.image_path,
            generation_kwargs=self.model.generation_kwargs,
            generation_mode=generation_mode,
            selected_images=selected_image
        )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    executor = ModelExecutor(cfg)
    executor()
    print("recorded result")


if __name__ == "__main__":
    main()
