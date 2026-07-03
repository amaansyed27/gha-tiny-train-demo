import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainingResult:
    epochs: int
    learning_rate: float
    weight: float
    bias: float
    final_loss: float
    loss_history: list[float]


def run_training(
    epochs: int = 10,
    learning_rate: float = 0.1,
    output_path: Path | None = None,
) -> TrainingResult:
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = 2.0 * x + 1.0

    weight = 0.0
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(epochs):
        predictions = weight * x + bias
        errors = predictions - y
        loss = float(np.mean(errors**2))
        loss_history.append(loss)

        grad_weight = float(2.0 * np.mean(errors * x))
        grad_bias = float(2.0 * np.mean(errors))

        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

    result = TrainingResult(
        epochs=epochs,
        learning_rate=learning_rate,
        weight=weight,
        bias=bias,
        final_loss=loss_history[-1],
        loss_history=loss_history,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny linear model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Where to write training metrics.",
    )
    args = parser.parse_args()

    result = run_training(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_path=args.output,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
