# GitHub Actions Tiny Train Demo

This repo trains a toy linear model for 10 epochs so we can test a harmless GitHub Actions workflow on a public repository.

## What it does

- Runs `pytest` against a tiny gradient-descent trainer.
- Trains the model for 10 epochs on a four-point dataset.
- Uploads a `metrics.json` artifact from GitHub Actions.

## Local run

```bash
pip install -r requirements.txt
pytest -q
python -m tiny_train.train --epochs 10 --learning-rate 0.1 --output artifacts/metrics.json
```
