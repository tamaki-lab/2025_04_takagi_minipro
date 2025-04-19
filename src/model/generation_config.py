from dataclasses import dataclass, asdict


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
