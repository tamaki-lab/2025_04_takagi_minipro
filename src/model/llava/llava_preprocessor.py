from transformers import LlavaProcessor
import torch


class LlavaPreprocessor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda:2"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = LlavaProcessor.from_pretrained(model_name)

    def prepare_inputs(self, prompt, image):
        # processor による前処理
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.bfloat16)
        return inputs
