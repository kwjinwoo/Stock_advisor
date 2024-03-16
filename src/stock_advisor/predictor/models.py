import torch.nn as nn


class PredictorMaker:
    """Create stock predict model nn.Module"""

    @staticmethod
    def make_transforemr():
        pass


class TransformerPredictor(nn.Transformer):
    """Transformer block based predictor"""

    def __init__(self, embedding_dim: int = 512, **kwargs) -> None:
        super().__init__(d_model=embedding_dim, **kwargs)
