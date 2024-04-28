from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    """Naive Transformer config"""

    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-5
    norm_first: bool = True
    bias = False


class PredictorMaker:
    """Create stock predict model nn.Module"""

    @staticmethod
    def make_transforemr():
        pass


class TransformerPredictor(nn.Module):
    """Transformer block based predictor"""

    def __init__(self, config: TransformerConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embed = nn.Linear(in_features=1, out_features=config.d_model)
        self.encoders = self.make_encoders()
        self.decoders = self.make_decoders()

    def make_encoders(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm_eps=self.config.layer_norm_eps,
            batch_first=True,
            norm_first=self.config.norm_first,
            bias=self.config.bias,
        )
        encoders = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.config.num_encoder_layers
        )
        return encoders

    def make_decoders(self) -> nn.TransformerDecoder:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            layer_norm_eps=self.config.layer_norm_eps,
            batch_first=True,
            norm_first=self.config.norm_first,
            bias=self.config.bias,
        )
        decoders = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.config.num_decoder_layers
        )
        return decoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.encoders(x)
        # TODO(kwjinwoo): studying about decoder forward and encoder output.
        # out = self.decoders()
        return x


if __name__ == "__main__":
    config = TransformerConfig()
    m = TransformerPredictor(config)

    random_input = torch.randn(1, 15, 1)
    out = m(random_input)
    print(out.shape)
