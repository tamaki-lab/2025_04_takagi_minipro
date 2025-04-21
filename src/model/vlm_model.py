from PIL import Image
import hydra
from omegaconf import DictConfig
from .model_factory import configure_model
from .generation_config import GenerationConfig
from .record_result import record_result


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
            generation_mode=generation_mode
        )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    executor = ModelExecutor(cfg)
    executor()
    print("recorded result")


if __name__ == "__main__":
    main()
