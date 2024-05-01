import json
from pathlib import Path
from typing import Any, Dict, Union


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """from json file path, load data.

    Args:
        path (Union[str, Path]): json file path.

    Returns:
        Dict[str, Any]: json data.
    """
    with open(path) as f:
        data = json.load(f)
    return data
