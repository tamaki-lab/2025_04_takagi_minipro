import requests
from PIL import Image
import torch
import torch.nn as nn

from mantis.models.conversation import Conversation, SeparatorStyle
from mantis.models.mllava import chat_mllava, LlavaForConditionalGeneration, MLlavaProcessor
from mantis.models.mllava.utils import conv_templates


class Llama(nn.Module):
    def __init__(self, max_new_tokens=128):
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
            # "no_repeat_ngram_size": 3,
        }

    def prepare_model(self):
        # セッション用プロンプトの設定

        conv_llama_3_elyza = Conversation(
            system=(
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
            ),
            roles=("user", "assistant"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.LLAMA_3,
            sep="<|eot_id|>",
        )
        conv_templates["llama_3"] = conv_llama_3_elyza

        # プロセッサとモデルの読み込み
        self.processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = LlavaForConditionalGeneration.from_pretrained(
            "SakanaAI/Llama-3-EvoVLM-JP-v2",
            torch_dtype=torch.float16,
            device_map=self.device
        ).eval()

    def forward(
        self,
        prompt,
        image,
        history=None
    ):

        response, history = chat_mllava(
            prompt, image, self.model, self.processor, history=history, **self.generation_kwargs
        )
        return response, history
