---
title: Prefero
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Prefero

Prefero is a browser app and Python toolkit for estimating and comparing discrete choice models.

[Open the hosted demo](https://huggingface.co/spaces/Wil2200/prefero). If the Space is asleep, run Prefero locally or wait for its owner to restart it.

## Try it with sample data

1. Open **Data**, expand **Alternative: Generate or Load Sample Data**, and select **Generate Simulated Dataset**.
2. Open **Model**, keep **Conditional Logit** and the default attributes, then select **Run Estimation**.
3. Open **Results** and check convergence before interpreting estimates. The **Apollo** tab on the Data page also offers an observed route choice dataset.

## Bring your own data

Upload a CSV or Excel workbook (`.xlsx`). Use one row per alternative in each respondent's choice task, or convert a supported wide layout in the app. Download the Excel template from the Data page. Use the same number of alternatives in every task, with at least two alternatives, or at least three for best–worst estimation.

Read the [data, methods and privacy guide](docs/USER_GUIDE.md) for encoding examples, model limits and what the hosted app processes. Run a local instance for files you cannot share with a third-party host.

## Run locally

Use Python 3.12 from the repository directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements.txt
python -m streamlit run app/Home.py --server.address 127.0.0.1
```

Open [localhost:7860](http://localhost:7860). The Docker deployment uses the same pinned runtime requirements. For CUDA, ROCm or Apple MPS, see the [local hardware setup](docs/USER_GUIDE.md#local-hardware-setup).

## Run the CPU example

```bash
python scripts/generate_simulated_data.py --seed 42 --utility-columns
python scripts/run_estimation.py \
  --data data/generated/simulated_choice_data.csv \
  --id-col respondent_id --task-col task_id \
  --alt-col alternative --choice-col choice \
  --variables 'log1p_price:normal,time_10min:normal,comfort:normal,reliability:fixed,asc_alt_2:fixed,asc_alt_3:fixed' \
  --model mixed --draws 200 --maxiter 150 --device cpu --out-prefix outputs/sim_mxl
```

The example uses the simulator's utility columns. Look for `Converged: True`, then inspect the estimates CSV and summary JSON under `outputs/`. A fit that fails to converge exits with status 1 and writes diagnostic output.

## Project links

[Contributing](CONTRIBUTING.md) · [Issues](https://github.com/WilZ2200/prefero/issues) · [Licence](LICENSE)
