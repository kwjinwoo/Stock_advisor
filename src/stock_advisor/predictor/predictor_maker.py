from pathlib import Path
from typing import Union

from stock_advisor.predictor.models import TransformerConfig, TransformerPredictor
from stock_advisor.utils import load_json


class PredictorMaker:
    """Create stock predictor"""

    @staticmethod
    def make_transformer(config_path: Union[str, Path]) -> TransformerPredictor:
        """make transformer predictor

        Args:
            config_path (Union[str, Path]): config json file path.

        Returns:
            TransformerPredictor: transformer predictor instance
        """
        config = TransformerConfig(**load_json(config_path))
        model = TransformerPredictor(config)
        return model
