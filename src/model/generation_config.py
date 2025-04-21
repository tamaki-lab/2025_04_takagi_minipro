from dataclasses import dataclass, asdict
from omegaconf import DictConfig
from enum import Enum


class GenerationMode(str, Enum):
    GREEDY_SEARCH = "greedy"
    SAMPLE = "sampling"
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    GROUP_BEAM_SEARCH = "group_beam_search"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    CONTRASTIVE_SEARCH = "contrastive_search"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1000
    num_beams: int = 1
    num_beam_groups: int = 1
    do_sample: bool = True
    temperature: float = 0.2
    top_k: int = 4
    penalty_alpha: float = 0.6
    temperature: float = 0.2
    diversity_penalty: float = 0.0

    def to_dict(self):
        d = asdict(self)
        if not self.do_sample:
            d.pop("temperature", None)  # ← do_sample=Falseのときtemperatureを除外
        return d

    def get_generation_mode(self) -> str:
        if self.num_beams == 1:
            if not self.do_sample:
                if (
                    self.top_k is not None and self.top_k > 1
                    and self.penalty_alpha is not None and self.penalty_alpha > 0
                ):
                    return "contrastive_search"
                else:
                    return "greedy"
            else:
                return "sampling"
        else:
            if self.num_beam_groups > 1:
                return "group_beam_search"
            elif self.do_sample:
                return "beam_sample"
            else:
                return "beam_search"

    @staticmethod
    def from_cfg(cfg: DictConfig):
        return GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            num_beam_groups=cfg.get("num_beam_groups", 1),
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            penalty_alpha=cfg.penalty_alpha,
            diversity_penalty=cfg.diversity_penalty,
        )
