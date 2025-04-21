import os

from omegaconf import DictConfig

from model import (
    Llava,
    Llama,
    SmolVLM,
)
from .generation_config import GenerationConfig


def set_torch_home(
    cfg: DictConfig
) -> None:
    """Specity the directory where a pre-trained model is stored.
    Otherwise, by default, models are stored in users home dir `~/.torch`
    """
    os.environ['TORCH_HOME'] = cfg.torch_home


def configure_model(
        cfg: DictConfig,
        model_info: GenerationConfig
):
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    if cfg.use_pretrained:
        set_torch_home(cfg)

    if cfg.model_name == 'llava':
        model = Llava(generation_config=model_info)

    elif cfg.model_name == 'llama':
        model = Llama(generation_config=model_info)

    elif cfg.model_name == 'smolvlm':
        model = SmolVLM(generation_config=model_info)

    else:
        raise ValueError('invalid model_info.model_name')

    return model
