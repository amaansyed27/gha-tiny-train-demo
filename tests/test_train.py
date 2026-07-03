import json
from pathlib import Path

import numpy as np

from tiny_train.train import run_training


def test_run_training_reduces_loss_and_learns_expected_line():
    result = run_training(epochs=10, learning_rate=0.1)

    assert len(result.loss_history) == 10
    assert result.loss_history[-1] < result.loss_history[0]
    assert abs(result.weight - 2.0) < 0.35
    assert abs(result.bias - 1.0) < 0.35


def test_run_training_can_write_metrics_file(tmp_path: Path):
    output_path = tmp_path / "metrics.json"

    result = run_training(epochs=10, learning_rate=0.1, output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["epochs"] == 10
    assert np.isclose(saved["final_loss"], result.loss_history[-1])
    assert np.isclose(saved["weight"], result.weight)
    assert np.isclose(saved["bias"], result.bias)
