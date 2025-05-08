from PIL import Image
import hydra
from omegaconf import DictConfig
import gc
import torch
from model.model_factory import configure_model
from model.generation_config import GenerationConfig
from model.record_result import record_result


class ModelExecutor:
    def __init__(self, cfg: DictConfig, model_name: str):
        self.cfg = cfg.copy()
        self.cfg.model_name = model_name  # 動的に上書き
        self.model_info = GenerationConfig.from_cfg(self.cfg)
        self.model = configure_model(self.cfg, self.model_info)

    def __call__(self):
        image = [Image.open(self.cfg.image_path)]
        prompt = self.cfg.prompt[self.cfg.model_name]  # モデル別プロンプトを選択

        self.model.prepare_model()
        output = self.model.forward(prompt, image)

        result_text = output[0] if isinstance(output, tuple) else output
        generation_mode = self.model_info.get_generation_mode()

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
        # 明示的にGPUメモリを解放
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_names = ["llama", "llava", "smolvlm"]
    for name in model_names:
        print(f"\n=== Running model: {name} ===")
        executor = ModelExecutor(cfg, model_name=name)
        executor()
    print("\nAll models finished.")


if __name__ == "__main__":
    main()
