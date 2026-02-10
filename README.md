---
title: 分析侠 Prefero
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 分析侠 Prefero — DCE Analysis Tool

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

The app is a multi-page Streamlit application with five pages:

| Page | Description |
|------|-------------|
| **1. Data** | Load simulated data, Apollo datasets, or upload your own CSV (long or wide format with automatic conversion). Preview data, panel structure, and descriptive statistics. |
| **2. Model** | Configure column roles, select utility variables, set distribution assumptions (fixed/normal/lognormal), choose Conditional, Mixed, or Latent Class Logit, and run estimation. |
| **3. Results** | Inspect model fit metrics, parameter estimates with standard errors and significance, interactive coefficient plots with CIs, WTP analysis, and predicted vs actual choice shares. Export estimates, summaries, and HTML reports. |
| **4. Compare** | Compare multiple estimated models side-by-side with fit metrics, parameter tables, coefficient plots, and likelihood ratio tests. |
| **5. Bootstrap** | Non-parametric bootstrap inference by resampling individuals. Compare bootstrap SEs with asymptotic SEs. |

## 6) Data Format

Expected long format (one row per alternative):
- ID column (`respondent_id` style)
- task column (`task_id` style)
- alternative column (`alternative` style)
- choice column:
  - either one-hot (`0/1`, with one `1` per task), or
  - chosen alternative code repeated for all alternatives in the task
- numeric utility variables

