from pathlib import Path

from stock_advisor.predictor import PredictorMaker
from stock_advisor.predictor.models import TransformerPredictor


class TestPredictorMaker:
    """test predictor maker."""

    maker = PredictorMaker()

    def test_make_transformer(self, transformer_config_path: Path) -> None:
        """test make_transformer.

        Args:
            transformer_config_path (Path): transformer config path.
        """
        predictor = self.maker.make_transformer(transformer_config_path)
        assert isinstance(predictor, TransformerPredictor)
