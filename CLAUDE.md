# Prefero development guide

Prefero combines a Python estimation package with a Streamlit app and command-line examples. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for Python 3.12 setup, test commands and release guidance.

The canonical app entry point is `app/Home.py`. Keep estimation logic in `src/dce_analyzer/` and use `ChoiceTensors` as the model input. Convert supported wide data to long format before estimating. Keep CLI scripts thin and preserve older supported entry points.

Conditional logit, mixed logit and GMNL share the PyTorch optimiser. Latent class uses dedicated estimators. Changes to likelihoods or parameter layouts should preserve the correspondence between public model specifications, saved coefficients, inference and predictions.
