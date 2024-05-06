import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from stock_advisor.predictor.configs import DataConfig


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
    bias: bool = False


class PositionalEmbedding(nn.Module):
    """Postional Embedding module class"""

    def __init__(self, data_config: DataConfig, model_config: TransformerConfig, **kwargs) -> None:
        """init Postional Embedding

        Args:
            data_config (DataConfig): data config
            model_config (TransformerConfig): model config
        """
        super().__init__(**kwargs)
        self.max_len = data_config.max_len
        self.d_model = model_config.d_model
        self.position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, self.max_len, self.d_model)
        pe[..., 0::2] = torch.sin(self.position * div_term)
        pe[..., 1::2] = torch.cos(self.position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwarding positional embedding. add position value to embedding vector.

        Args:
            x (torch.Tensor): embedding vector.

        Returns:
            torch.Tensor: embedding vector to be added postion info.
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerPredictor(nn.Module):
    """Transformer block based predictor"""

    def __init__(self, data_config: DataConfig, model_config: TransformerConfig, **kwargs) -> None:
        """init TransformerPredictor

        Args:
            data_config(DataConfig): Data config
            model_config (TransformerConfig): Transformer config
        """
        super().__init__(**kwargs)
        self.config = model_config
        self.pe = PositionalEmbedding(data_config=data_config, model_config=model_config)
        self.embed = nn.Linear(in_features=1, out_features=model_config.d_model)
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
        encoders = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.config.num_encoder_layers)
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
        decoders = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.config.num_decoder_layers)
        return decoders

    def forward(self, inputs: torch.Tensor, decoder_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward. if decoder_input is None, encoder output's last value is passed to decoder's tgt.
        except when decoder_input len is one, always causal mask is applied

        Args:
            inputs (torch.Tensor): input data
            decoder_input (Optional[torch.Tensor], optional): decoder input data. Defaults to None.

        Returns:
            torch.Tensor: predict values.
        """
        encoder_input = self.embed(inputs)
        encoder_input = self.pe(encoder_input)
        encoder_output = self.encoders(encoder_input)

        if decoder_input is None:
            decoder_input = encoder_output[..., [-1], :]
            encoder_output = encoder_output[..., :-1, :]
        else:
            decoder_input = self.embed(decoder_input)

        if decoder_input.shape[1] > 1:
            mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1])
        else:
            mask = None
        decoder_input = self.pe(decoder_input)
        out = self.decoders(tgt=decoder_input, memory=encoder_input, tgt_mask=mask)
        return out
