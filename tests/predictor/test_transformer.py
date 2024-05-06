from typing import List, Optional

import pytest
import torch

from stock_advisor.predictor.configs import DataConfig
from stock_advisor.predictor.models import TransformerConfig, TransformerPredictor


class TestTransformerPredictor:
    """TransformerPredictor class tests."""

    data_config = DataConfig()
    model_config = TransformerConfig()
    transformer = TransformerPredictor(data_config=data_config, model_config=model_config)

    def test_encoder(self) -> None:
        """encdoer test"""
        inputs = torch.randn(1, 14, self.model_config.d_model)

        encoder = self.transformer.encoders
        out = encoder(inputs)

        assert out.shape == inputs.shape

    def test_decoder(self) -> None:
        """decoder test"""
        encoder_out = torch.randn(1, 14, self.model_config.d_model)
        inputs = torch.randn(1, 1, self.model_config.d_model)

        decoder = self.transformer.decoders
        out = decoder(inputs, encoder_out)
        assert out.shape == inputs.shape

    @pytest.mark.parametrize(
        "encoder_input, decoder_input",
        (
            [torch.randn(1, 14, 1), torch.randn(1, 14, 1)],
            [torch.randn(1, 14, 1), torch.randn(1, 1, 1)],
            [torch.randn(1, 14, 1), None],
        ),
    )
    def test_forward(self, encoder_input: torch.Tensor, decoder_input: Optional[torch.Tensor]) -> None:
        """forward test"""
        out = self.transformer(encoder_input, decoder_input)

        if decoder_input is not None:
            answer_shape = list(decoder_input.shape)
        else:
            answer_shape = list(encoder_input[..., [-1], :].shape)
        answer_shape[-1] = self.model_config.d_model
        assert list(out.shape) == answer_shape

    @pytest.mark.parametrize("input_shape", [([1, 1, 512]), ([1, 14, 512])])
    def test_pe(self, input_shape: List[int]) -> None:
        """test Postional Embeeding

        Args:
            input_shape (List[int]): input shape
        """
        temp_input = torch.randn(input_shape)
        pe = self.transformer.pe

        out = pe(temp_input)

        assert out.shape == temp_input.shape
