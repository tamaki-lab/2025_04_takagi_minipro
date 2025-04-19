import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from .llava_preprocessor import LlavaPreprocessor
from ..generation_config import GenerationConfig


class Llava(nn.Module):
    def __init__(self, generation_config: GenerationConfig = GenerationConfig()):
        super().__init__()
        self.device = "cuda:2" if torch.cuda.device_count() >= 3 else "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = generation_config.to_dict()

    def prepare_model(self):
        self.processor = LlavaPreprocessor(device=self.device)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16
        ).to(self.device).eval()

    def forward(self, prompt, image):
        inputs = self.processor.prepare_inputs(prompt, image)
        generate_ids = self.model.generate(**inputs, **self.generation_kwargs)
        output = self.processor.processor.tokenizer.decode(
            generate_ids[0][:-1],
            clean_up_tokenization_spaces=False
        )
        return output
