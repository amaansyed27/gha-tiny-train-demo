# GitHub Actions Tiny Train Demo

This repo trains a toy linear model for 10 epochs so we can test a harmless GitHub Actions workflow on a public repository.

## What it does

- Runs `pytest` against a tiny gradient-descent trainer.
- Trains the model for 10 epochs on a four-point dataset.
- Uploads a `metrics.json` artifact from GitHub Actions.
- Lets you choose the epoch count when you start a manual Actions run.

## Local run

```bash
pip install -r requirements.txt
pytest -q
python -m tiny_train.train --epochs 10 --learning-rate 0.1 --output artifacts/metrics.json
```

## Manual GitHub Actions run

1. Open the repo's `Actions` tab.
2. Open the `Tiny Train` workflow.
3. Click `Run workflow`.
4. Enter an epoch count such as `10`, `25`, or `50`.
5. Download the `tiny-train-metrics` artifact after the run finishes.

## What you can download and use

This demo does not produce a reusable neural-network checkpoint or a packaged model file. It produces a tiny metrics JSON artifact with:

- the epoch count used
- the learned `weight` and `bias`
- the loss history
- the final loss

You can use it in two simple ways:

1. Download `tiny-train-metrics` from the workflow run and inspect the learned parameters.
2. Run the trainer locally with your own epoch count to regenerate the same kind of output:

```bash
python -m tiny_train.train --epochs 25 --learning-rate 0.1 --output artifacts/metrics.json
```

For this toy example, the "model" is just the line:

```text
y = weight * x + bias
```

So if the artifact says `weight` is about `2.01` and `bias` is about `0.97`, you can plug those directly into any tiny script that needs the fitted line.
