import json
from pathlib import Path

import pytest


@pytest.fixture
def transformer_config_path(tmp_path: Path) -> str:
    temp_config = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "norm_first": True,
        "bias": False,
    }
    save_path = tmp_path / "transformer.json"

    with open(save_path, "w") as f:
        json.dump(temp_config, f)
    return save_path


@pytest.fixture
def data_config_path(tmp_path: Path) -> str:
    temp_config = {"max_len": 14}
    save_path = tmp_path / "data.json"
    with open(save_path, "w") as f:
        json.dump(temp_config, f)
    return save_path
