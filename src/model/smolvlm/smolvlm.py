# model/smolvlm.py
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq
from .smolvlm_preprocessor import SmolVLMPreprocessor
from ..generation_config import GenerationConfig


class SmolVLM(nn.Module):
    def __init__(self, generation_config: GenerationConfig = GenerationConfig()):
        super().__init__()
        self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = generation_config.to_dict()

    def prepare_model(self):
        # self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.processor = SmolVLMPreprocessor(device=self.device)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager"
        ).to(self.device).eval()

    def forward(self, prompt, image):
        inputs = self.processor.prepare_inputs(prompt, image)
        generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
        generated_texts = self.processor.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts[0]
