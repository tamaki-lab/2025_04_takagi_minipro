import torch
import torch.nn as nn

from ..generation_config import GenerationConfig
from mantis.models.conversation import Conversation, SeparatorStyle
from mantis.models.mllava import chat_mllava, LlavaForConditionalGeneration
from mantis.models.mllava.utils import conv_templates

from .llama_preprocessor import LlamaPreprocessor  # ★ 追加


class Llama(nn.Module):
    def __init__(self, generation_config: GenerationConfig = GenerationConfig()):
        super().__init__()
        self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = generation_config.to_dict()

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

        # ★ ここでラッパーを使用
        self.processor = LlamaPreprocessor(device=self.device)

        # モデルの読み込み
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
        # ★ processor ラッパーの中の processor を使う
        response, history = chat_mllava(
            prompt,
            image,
            self.model,
            self.processor.processor,  # ★ 修正ポイント
            history=history,
            **self.generation_kwargs
        )
        return response, history
