# Data, methods and privacy

## Prepare choice data

Provide respondent ID, task ID, alternative ID, choice, and one or more attributes. Each respondent, task and alternative combination must be unique. Use numeric attribute columns and keep the four structural columns complete. A choice column can contain one 1 per task with 0 on all other rows. It can also repeat the selected alternative's ID on every row of a task.

| respondent_id | task_id | alternative | choice | price |
| --- | --- | --- | --- | --- |
| 10 | 1 | A | 1 | 5 |
| 10 | 1 | B | 0 | 7 |
| 10 | 2 | A | 0 | 9 |
| 10 | 2 | B | 1 | 4 |

For repeated alternative IDs, the choice column in this example would contain A, A, B, B. See the Data page for the input template, supported wide layouts, and optional Excel data dictionaries. To use an older `.xls` workbook, save it as `.xlsx` or export the data sheet as CSV.

For best–worst tasks, add a worst-choice column using either encoding. Every task must contain at least three alternatives and different best and worst choices. Respondent covariates for class membership should be constant across that respondent's rows.

## Interpret a fit

Conditional logit is a baseline with common preferences. Mixed logit models random coefficients, GMNL adds scale heterogeneity, and latent class estimates discrete preference segments. Conditional logit, mixed logit and GMNL use PyTorch L-BFGS, with convergence checked on the internally scaled objective. Estimation reports coefficients in the input units. Mixed models use Halton draws by default. A converged run can still lack covariance estimates if the fitted Hessian is not positive definite.

Use the Results page to review convergence, inference and predicted shares. Use Compare to evaluate prediction on held-out respondents. For bootstrap inference, check how many replications converged before interpreting the intervals. Willingness to pay requires a clearly defined cost variable.

## Local hardware setup

The hosted demo runs on CPU. For local GPU use, install the PyTorch wheel appropriate to your hardware using [PyTorch's installation guide](https://pytorch.org/get-started/locally/), then install the app with `python -m pip install -e .`. Prefero automatically uses CUDA (also the backend name on ROCm) or Apple MPS when available. Check the device on the Model page. The CLI accepts `--device cpu` for a CPU run.

## Data handling

The browser sends uploads to the server running Prefero. Prefero holds uploaded data and model results in the analysis session. Download your results before closing the session. The Community page links to public feedback channels. Use synthetic inputs in public issues.

Telemetry is disabled by default. When a self-hosted operator sets `PREFERO_TELEMETRY_ENABLED=true`, Prefero shows a notice and writes coarse event types and timestamps to local SQLite. Set `PREFERO_DB_PATH` to choose the database location. Prefero makes no automatic remote telemetry transfers or IP geolocation requests.
