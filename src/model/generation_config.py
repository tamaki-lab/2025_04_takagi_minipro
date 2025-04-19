from dataclasses import dataclass, asdict
from omegaconf import DictConfig


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1000
    num_beams: int = 1
    do_sample: bool = True
    temperature: float = 0.2
    top_k: int = 4
    penalty_alpha: float = 0.6
    temperature: float = 0.2

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_cfg(cfg: DictConfig):
        return GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            penalty_alpha=cfg.penalty_alpha,
        )
