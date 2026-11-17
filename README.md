---
title: Prefero

---

# Prefero

Prefero is a browser app for analysing discrete choice experiments. You can see how people trade off things like price, travel time and comfort, then compare models and download your results.

**[Try Prefero in your browser](https://huggingface.co/spaces/Wil2200/prefero)**

## Try it with sample data

You can try a model before uploading your own data. Open **Data**, expand **Alternative: Generate or Load Sample Data**, and select **Generate Simulated Dataset**. Then open **Model**, keep **Conditional Logit** and the default attributes, and click **Run Estimation**. Open **Results** to see the estimates and check whether the model converged.

If you’d prefer an example with observed choices, pick a dataset from the **Apollo** tab in the same panel.

## Bring your own data

Upload a CSV or Excel file on the **Data** page. Prefero fits models using long format, with one row per alternative in each respondent’s choice task. You’ll need columns for the respondent, task, alternative, choice and attributes. The choice column can mark the selected row with a 1, or repeat the selected alternative’s code across the task. The app can help map your columns and convert supported wide layouts.

Start with conditional logit for a baseline. If you want to see how preferences vary across respondents, try mixed logit, GMNL or latent class. You can also work with best–worst data, compare model fit, estimate willingness to pay, bootstrap by respondent and export an HTML report.

## Run Prefero locally

From the repository directory, use Python 3.12 to create an environment and start the app.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

Open [localhost:7860](http://localhost:7860) in your browser.

For a command-line run, generate a dataset and fit a mixed logit model.

```bash
python scripts/generate_simulated_data.py --n-individuals 300 --n-tasks 8 --n-alts 3 --seed 42
python scripts/run_estimation.py \
  --data data/generated/simulated_choice_data.csv \
  --id-col respondent_id --task-col task_id \
  --alt-col alternative --choice-col choice \
  --variables "price:normal,time:normal,comfort:fixed,reliability:fixed" \
  --model mixed --draws 200 --maxiter 150 --out-prefix outputs/sim_mxl
```

Look in `outputs/` for the estimates CSV and model summary JSON. To download the Apollo route choice example locally, run `python scripts/fetch_apollo_data.py --dataset swiss_route_choice`.

## A few details about estimation

Prefero uses Halton draws in the app. Conditional logit, mixed logit and GMNL use PyTorch L-BFGS and a maximum per-task gradient tolerance of `1e-4` in FP32. Latent class uses separate EM and direct estimators. If the Hessian is not positive definite, Prefero leaves covariance estimates unavailable. Bootstrap summaries count only converged, finite fits.

### Using your GPU

When you run Prefero locally, it automatically uses a GPU if PyTorch can see one. That includes Apple Silicon via MPS, supported AMD GPUs via ROCm, and NVIDIA GPUs via CUDA. Otherwise, Prefero uses the CPU, as the hosted demo does.

The **Model** page shows which device Prefero is using. On AMD, you may see the label **CUDA** because PyTorch uses the same interface for ROCm. If your GPU isn’t detected, check that you’ve installed the [PyTorch build for your hardware](https://pytorch.org/get-started/locally/).

## Questions or feedback?

If you run into trouble, [open an issue](https://github.com/WilZ2200/prefero/issues). Tell us what you were trying to do and what happened. A small synthetic example helps if you can share one.

See the [licence](LICENSE) for AGPL and commercial terms.
