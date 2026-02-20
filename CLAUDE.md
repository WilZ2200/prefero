# Prefero — GPU-Enabled DCE Analysis Tool

## Project Overview

A GPU-accelerated discrete choice experiment (DCE) analysis toolkit for mixed logit estimation, built with PyTorch. This project combines a reusable Python library, interactive Streamlit app, and CLI tools.

## Tech Stack

- **Language:** Python 3.10+
- **Core:** PyTorch (GPU/autodiff), scipy (L-BFGS-B optimizer), numpy, pandas
- **UI:** Streamlit
- **Package:** setuptools with `src/` layout

## Project Structure

```
src/dce_analyzer/       # Core estimation engine (Python package)
  config.py             # ModelSpec, VariableSpec dataclasses
  data.py               # Long-format data loading & validation → ChoiceTensors
  model.py              # MixedLogitEstimator, ConditionalLogitEstimator
  latent_class.py       # LatentClassEstimator (Q discrete segments, softmax class probs)
  pipeline.py           # estimate_dataframe() orchestration & output saving
  simulate.py           # generate_simulated_dce() synthetic data generator
  apollo.py             # Apollo dataset downloader & long-format converter

app/streamlit_app.py    # Interactive Streamlit web UI
scripts/                # CLI entry points (generate data, fetch Apollo, run estimation)
data/                   # raw/, generated/, processed/ datasets
outputs/                # Estimation results (CSV + JSON)
```

## Key Concepts

- **Long-format data:** One row per alternative per task per individual
- **Mixed logit:** Random coefficients with fixed/normal/lognormal distributions
- **Halton sequences:** Quasi-Monte Carlo draws for simulation

## Development Guidelines

- All estimation code lives in `src/dce_analyzer/` — keep it modular
- Data must always be in long format before estimation
- Use `ChoiceTensors` as the standard internal data representation
- PyTorch tensors for all numerical computation (enables GPU + autodiff)
- scipy L-BFGS-B for optimization (wraps PyTorch gradient computation)
- Follow existing patterns in model.py for new estimator variants
- Scripts in `scripts/` are thin CLI wrappers around library functions

## Running the Project

```bash
# Setup
pip install -e .

# Generate simulated data
python scripts/generate_simulated_data.py --n-individuals 300 --n-tasks 8 --n-alts 3

# Fetch real Apollo data
python scripts/fetch_apollo_data.py --dataset swiss_route_choice

# Run estimation
python scripts/run_estimation.py \
  --data data/generated/simulated_choice_data.csv \
  --id-col respondent_id --task-col task_id --alt-col alternative --choice-col choice \
  --variables "price:normal,time:normal,comfort:fixed" \
  --model mixed --draws 200

# Streamlit app
streamlit run app/streamlit_app.py
```

