from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

from stock_advisor.predictor.configs import TrainConfig
from stock_advisor.utils import load_json

OPTIMIZER_MAP = {
    "Adam": Adam,
    "SGD": SGD,
}

LOSS_FN_MAP = {
    "MAE": nn.L1Loss,
}


class Trainer:
    def __init__(self, config_path: str | Path, model: nn.Module, data_loader: DataLoader) -> None:
        """initialize Trainer.

        Args:
            config_path (str | Path): train config path.
            model (nn.Module): model to train.
        """
        self.config = TrainConfig(**load_json(config_path))
        self.model = model
        self.data_loader = data_loader
        self.optimizer = self.get_optimizer()
        self.loss_fn = self.get_loss_fn()

    def get_optimizer(self) -> Optimizer:
        """get optimizer from config.

        Returns:
            Optimizer: optimizer.
        """
        optimizer = OPTIMIZER_MAP[self.config.optimizer]
        return optimizer(self.model.parameters(), **self.config.get_optimizer_configs())

    def get_loss_fn(self) -> nn.modules.loss._Loss:
        """get loss function from configs.

        Returns:
            nn.modules.loss._Loss: loss function
        """
        return LOSS_FN_MAP[self.config.loss_fn]()

    def run_train(self) -> None:
        """train model."""
        for epoch in self.config.num_epochs:
            batch_loss = self.run_epoch()
            print(f"{epoch} loss: {batch_loss / len(self.data_loader)}")

    def run_epoch(self) -> torch.Tensor:
        """run epoch

        Returns:
            torch.Tensor: one epoch mean loss
        """
        batch_loss = 0.0
        for batch in self.data_loader:
            batch_loss += self.run_batch(*batch)
        return batch_loss / len(self.data_loader)

    def run_batch(self, batch: Tuple[torch.Tensor]) -> torch.Tensor:
        """run batch.

        Args:
            batch (Tuple[torch.Tensor]): batch value.

        Returns:
            torch.Tensor: batch loss.
        """
        inputs, dec_inputs, outputs = batch
        self.optimizer.zero_grad()

        results = self.model(inputs, dec_inputs)
        loss = self.loss_fn(results, outputs)

        loss.backward()
        self.optimizer.step()

        return loss
