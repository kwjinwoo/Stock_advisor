from pathlib import Path
from typing import Union

from stock_advisor.predictor.configs import DataConfig
from stock_advisor.predictor.models import TransformerConfig, TransformerPredictor
from stock_advisor.utils import load_json


class PredictorMaker:
    """Create stock predictor"""

    @staticmethod
    def make_transformer(
        data_config_path: Union[str, Path], model_config_path: Union[str, Path]
    ) -> TransformerPredictor:
        """make transformer predictor

        Args:
            config_path (Union[str, Path]): config json file path.

        Returns:
            TransformerPredictor: transformer predictor instance
        """
        data_config = DataConfig(**load_json(data_config_path))
        model_config = TransformerConfig(**load_json(model_config_path))
        model = TransformerPredictor(data_config=data_config, model_config=model_config)
        return model
