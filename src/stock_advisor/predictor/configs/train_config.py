from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainConfig:
    """train config class"""

    batch_size: int = 4
    num_epochs: int = 100
    optimizer: str = "Adam"
    lr: float = 1e-5

    def get_optimizer_configs(self) -> Dict:
        """get optimizer arguments.

        Returns:
            Dict: optimizer's arguments dicts.
        """
        return {
            "lr": self.lr,
        }
