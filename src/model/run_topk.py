import hydra
from copy import deepcopy
from .vlm_model import ModelExecutor


@hydra.main(config_path="../../config", config_name="llava", version_base=None)
def run_all(cfg):
    for k in range(1, 21):
        cfg_loop = deepcopy(cfg)
        cfg_loop.top_k = k
        print(f"\n=== Running with top_k={k} ===")
        executor = ModelExecutor(cfg_loop)
        executor()


if __name__ == "__main__":
    run_all()
