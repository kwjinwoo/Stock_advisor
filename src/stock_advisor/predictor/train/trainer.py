from pathlib import Path

import torch.nn as nn
from torch.optim import SGD, Adam, Optimizer

from stock_advisor.predictor.configs import TrainConfig
from stock_advisor.utils import load_json

OPTIMIZER_MAP = {
    "Adam": Adam,
    "SGD": SGD,
}


class Trainer:
    def __init__(self, config_path: str | Path, model: nn.Module) -> None:
        """initiate Trainer.

        Args:
            config_path (str | Path): train config path.
            model (nn.Module): model to train.
        """
        self.config = TrainConfig(**load_json(config_path))
        self.model = model
        self.optimizer = self.get_optimizer()

    def get_optimizer(self) -> Optimizer:
        """get optimizer from config.

        Returns:
            Optimizer: optimizer.
        """
        optimizer = OPTIMIZER_MAP[self.config.optimizer]
        return optimizer(self.model.parameters(), **self.config.get_optimizer_configs())
