from typing import Optional

import pytest
import torch

from stock_advisor.predictor.models import TransformerConfig, TransformerPredictor


class TestTransformerPredictor:
    """TransformerPredictor class tests."""

    config = TransformerConfig()
    transformer = TransformerPredictor(config)

    def test_encoder(self) -> None:
        """encdoer test"""
        inputs = torch.randn(1, 15, self.config.d_model)

        encoder = self.transformer.encoders
        out = encoder(inputs)

        assert out.shape == inputs.shape

    def test_decoder(self) -> None:
        """decoder test"""
        encoder_out = torch.randn(1, 15, self.config.d_model)
        inputs = torch.randn(1, 1, self.config.d_model)

        decoder = self.transformer.decoders
        out = decoder(inputs, encoder_out)
        assert out.shape == inputs.shape

    @pytest.mark.parametrize(
        "encoder_input, decoder_input",
        (
            [torch.randn(1, 15, 1), torch.randn(1, 15, 1)],
            [torch.randn(1, 15, 1), torch.randn(1, 1, 1)],
            [torch.randn(1, 15, 1), None],
        ),
    )
    def test_forward(
        self, encoder_input: torch.Tensor, decoder_input: Optional[torch.Tensor]
    ) -> None:
        """forward test"""
        out = self.transformer(encoder_input, decoder_input)

        if decoder_input is not None:
            answer_shape = list(decoder_input.shape)
        else:
            answer_shape = list(encoder_input[..., [-1], :].shape)
        answer_shape[-1] = self.config.d_model
        assert list(out.shape) == answer_shape
