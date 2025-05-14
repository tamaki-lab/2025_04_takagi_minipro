import torch
from mantis.models.mllava import MLlavaProcessor


class LlamaPreprocessor:
    def __init__(self, model_name="TIGER-Lab/Mantis-8B-siglip-llama3", device="cuda:3"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = MLlavaProcessor.from_pretrained(model_name)
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def prepare_inputs(self, prompt, image):
        # LLaVAと同様に processor で画像とテキストを処理
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device, torch.bfloat16)
        return inputs
