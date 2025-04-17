import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image


class Llava(nn.Module):
    def __init__(self, max_new_tokens=1000):
        super().__init__()
        self.device = "cuda:2" if torch.cuda.device_count() >= 3 else "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,  # greedy
            # "num_beams": 2,  # beam
            # "do_sample": False,
            "do_sample": True,  # beam_sample

            # "top_k": 4,  # CONTRASTIVE_SEARCH
            # "penalty_alpha": 0.6,
            "temperature": 0.2,
        }

    def prepare_model(self):
        # モデルとプロセッサの準備
        self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16
        ).to(self.device).eval()

    def forward(
        self,
        prompt,
        image
    ):
        # 入力変換
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.bfloat16)

        # 推論
        generate_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs
        )

        # 出力をデコード
        output = self.processor.tokenizer.decode(
            generate_ids[0][:-1],
            clean_up_tokenization_spaces=False
        )
        return output
