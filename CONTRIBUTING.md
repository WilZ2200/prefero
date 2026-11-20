# Contributing to Prefero

Use Python 3.12. From a fresh checkout, follow the [local setup](README.md#run-locally), then install the development tools with `python -m pip install -e '.[dev]' -r requirements.txt`.

```bash
python scripts/test_e2e.py
python scripts/test_cv.py
python scripts/test_new_modules.py
python -m pytest tests -q
```

When changing an estimator, compare the likelihood, parameter estimates and convergence status with an independent reference. Use fixed seeds and synthetic data for regressions.

The Streamlit app starts at `app/Home.py`. The `src/dce_analyzer` package contains the estimators, model specifications and data conversion. Scripts wrap the same package. Run `python scripts/run_estimation.py --help` to see the CLI options.

Use synthetic inputs when filing an issue. Include the Prefero commit, Python version, PyTorch version and device. Describe the requested result and attach a minimal example. See the [release guide](docs/RELEASING.md) for deployment checks and rollback.
