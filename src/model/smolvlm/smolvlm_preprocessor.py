from transformers import AutoProcessor
import torch


class SmolVLMPreprocessor:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-256M-Instruct", device="cuda:2"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)

    def prepare_inputs(self, prompt, image):
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Process inputs
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt").to(self.device)
        return inputs
