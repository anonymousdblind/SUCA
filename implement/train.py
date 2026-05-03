"""
SUCA Training Entry Point.

Usage:
    python train.py
    python train.py suca.tau=0.05 training.learning_rate=5e-7
    python train.py model.pretrained_model_name=runwayml/stable-diffusion-v1-5
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from suca.suca_trainer import SUCATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    repo_root = Path(__file__).resolve().parent
    if not cfg.logging.get("analysis_dir"):
        cfg.logging.analysis_dir = str((repo_root / "analysis").resolve())

    logger.info("SUCA — Semantic Unit Credit Assignment for Diffusion RL")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = SUCATrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
