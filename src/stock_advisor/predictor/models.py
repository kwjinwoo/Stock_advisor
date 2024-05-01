from dataclasses import dataclass
from typing import Optional

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
    """Create stock nn.Module predictor"""

    @staticmethod
    def make_transforemr():
        pass


class TransformerPredictor(nn.Module):
    """Transformer block based predictor"""

    def __init__(self, config: TransformerConfig, **kwargs) -> None:
        """init TransformerPredictor

        Args:
            config (TransformerConfig): Transformer config
        """
        super().__init__(**kwargs)
        self.config = config
        self.embed = nn.Linear(in_features=1, out_features=config.d_model)
        self.encoders = self.make_encoders()
        self.decoders = self.make_decoders()

    def make_encoders(self) -> nn.TransformerEncoder:
        """make encoder layers

        Returns:
            nn.TransformerEncoder: encoder layers
        """
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
        """make decoder layers

        Returns:
            nn.TransformerDecoder: decoder layers
        """
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

    def forward(
        self, inputs: torch.Tensor, decoder_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """forward. if decoder_input is None, encoder output's last value is passed to decoder's tgt.
        except when decoder_input len is one, always causal mask is applied

        Args:
            inputs (torch.Tensor): input data
            decoder_input (Optional[torch.Tensor], optional): decoder input data. Defaults to None.

        Returns:
            torch.Tensor: predict values.
        """
        encoder_input = self.embed(inputs)
        encoder_output = self.encoders(encoder_input)

        if decoder_input is None:
            decoder_input = encoder_output[..., [-1], :]
            encoder_output = encoder_output[..., :-1, :]
        else:
            decoder_input = self.embed(decoder_input)

        if decoder_input.shape[1] > 1:
            mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.shape[1]
            )
        else:
            mask = None

        out = self.decoders(tgt=decoder_input, memory=encoder_input, tgt_mask=mask)
        return out
