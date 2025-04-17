# model/smolvlm.py
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class SmolVLM(nn.Module):
    def __init__(self, max_new_tokens=500):
        super().__init__()
        self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,  # greedy
            # "num_beams": 2,  # beam
            # "do_sample": False,
            "do_sample": True,  # beam_sample
            "temperature": 0.2,

            # "top_k": 4,  # CONTRASTIVE_SEARCH
            # "penalty_alpha": 0.6,
        }

    def prepare_model(self):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager"
        ).to(self.device).eval()

    def forward(self, prompt, image):
        # create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts[0]
