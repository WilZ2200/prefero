---
title: Prefero
sdk: docker
app_port: 7860
---

# Prefero

Prefero is a browser app and Python toolkit for estimating and comparing discrete choice models. [Try the hosted demo](https://huggingface.co/spaces/Wil2200/prefero).

## Run locally

Use Python 3.12 and run these commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements.txt
python -m streamlit run app/Home.py --server.address 127.0.0.1
```

Open [localhost:7860](http://localhost:7860). Generate a sample dataset on **Data**, run conditional logit on **Model**, then check convergence on **Results**.

[User guide and CLI example](docs/USER_GUIDE.md) · [Contributing](CONTRIBUTING.md) · [Licence](LICENSE)
