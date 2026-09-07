---
title: Prefero
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Prefero — DCE Analysis Tool

End-to-end discrete choice experiment workflow:
- reusable estimation engine (`conditional logit` and `mixed logit`)
- Streamlit app for interactive runs
- simulated dataset generator
- Apollo discrete-choice data downloader and converter

Scope note: this project is focused on **long-format discrete-choice data**.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## 2) Generate Simulated Data

```bash
python scripts/generate_simulated_data.py \
  --n-individuals 300 \
  --n-tasks 8 \
  --n-alts 3 \
  --seed 42
```

Outputs:
- `data/generated/simulated_choice_data.csv`
- `data/generated/simulated_truth.json`

## 3) Use Real Apollo Data

```bash
python scripts/fetch_apollo_data.py --dataset swiss_route_choice
```

Outputs:
- raw CSV in `data/raw/`
- converted long-format CSV in `data/processed/`

## 4) Run Estimation from CLI

Example on simulated data:

```bash
python scripts/run_estimation.py \
  --data data/generated/simulated_choice_data.csv \
  --id-col respondent_id \
  --task-col task_id \
  --alt-col alternative \
  --choice-col choice \
  --variables "price:normal,time:normal,comfort:fixed,reliability:fixed" \
  --model mixed \
  --draws 200 \
  --maxiter 150 \
  --out-prefix outputs/sim_mxl
```

Outputs:
- `outputs/sim_mxl_estimates.csv`
- `outputs/sim_mxl_summary.json`

## 5) Run App

```bash
streamlit run app/Home.py
```

The app retains its Streamlit multipage workflow:

| Page | Capability |
|---|---|
| Data | Upload CSV/Excel, convert wide data, simulate DCE/BWS data, or load Apollo examples. |
| Model | Conditional Logit, Mixed Logit, GMNL and Latent Class; dummy coding, interactions, normal/lognormal coefficients, full/selective correlation and BWS. |
| Results | Model fit, convergence, estimates, covariance/correlation, WTP and predictions. |
| Compare | Model comparisons, statistical tests and respondent-level cross-validation. |
| Bootstrap | Respondent-level resampling with explicit successful replication counts. |
| Report | Download analysis reports and estimation outputs. |
| Community / Admin | Existing user and discussion workflows. |

### Estimation implementation (0.2)

CL, MXL and GMNL share a tensor-native PyTorch L-BFGS optimizer with strong-Wolfe
line search. The optimizer evaluates mean negative log-likelihood per choice task
and uses a maximum absolute per-task gradient tolerance of `1e-4` in FP32. Small
steps without meeting this gradient tolerance are reported as non-convergence.
The likelihood, parameter ordering, starting-vector format, results, saved-model
format, correlation structures and BWS definitions remain compatible. Latent
Class retains its dedicated EM/direct estimators. Covariance inference is omitted
when the Hessian is not positive definite; no diagonal shift is used to manufacture
standard errors. Bootstrap counts only converged, finite replications.

The public HF Space runs on CPU. Local runs can use supported CUDA or Apple MPS
hardware. The interactive app uses Halton draws.

### Verification and deployment

```bash
python scripts/test_e2e.py
python scripts/test_cv.py
python scripts/test_new_modules.py
python -m pytest tests -q
```

GitHub Actions runs these checks, builds the Docker image and verifies its health
endpoint before deploying `main`. HF updates must be fast-forwards; remote-only
changes stop deployment. Each deployment saves the prior HF commit under a
`releases/pre-<run-id>` branch. Roll back with an ordinary revert of the release
commit and the same verified deployment flow.

Runtime SQLite files and generated outputs are excluded from the application
image. Existing hosted account and forum persistence uses the configured HF
dataset. For isolated local tests, set `PREFERO_DB_PATH` to a temporary database
and do not supply HF credentials.

## 6) Data Format

Expected long format (one row per alternative):
- ID column (`respondent_id` style)
- task column (`task_id` style)
- alternative column (`alternative` style)
- choice column:
  - either one-hot (`0/1`, with one `1` per task), or
  - chosen alternative code repeated for all alternatives in the task
- numeric utility variables

