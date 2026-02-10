from __future__ import annotations

from pathlib import Path
import ssl
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


APOLLO_DATASETS = {
    "swiss_route_choice": "https://www.apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData.csv",
}

APOLLO_DATASET_INFO = {
    "swiss_route_choice": {
        "name": "Swiss Route Choice",
        "description": (
            "Route choice data from Axhausen et al. (2008). "
            "Respondents chose between 2 route alternatives across multiple choice tasks. "
            "Attributes: travel_time, travel_cost, headway, changes. "
            "Includes socio-demographic covariates."
        ),
        "n_respondents": 388,
        "n_tasks": 9,
        "n_alts": 2,
        "attributes": ["travel_time", "travel_cost", "headway", "changes"],
        "reference": "Axhausen et al. (2008)",
    },
}


def download_apollo_dataset(dataset: str, output_path: str | Path | None = None) -> pd.DataFrame:
    if dataset not in APOLLO_DATASETS:
        raise ValueError(f"Unsupported Apollo dataset '{dataset}'.")

    url = APOLLO_DATASETS[dataset]
    try:
        df = pd.read_csv(url)
    except (URLError, ssl.SSLError) as err:
        # Fallback for environments with missing local CA bundles.
        cert_error = "CERTIFICATE_VERIFY_FAILED" in str(err).upper()
        if not cert_error:
            raise
        insecure_ctx = ssl.create_default_context()
        insecure_ctx.check_hostname = False
        insecure_ctx.verify_mode = ssl.CERT_NONE
        with urlopen(url, context=insecure_ctx) as response:
            df = pd.read_csv(response)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def _to_long_swiss_route(df: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", "choice", "tt1", "tc1", "hw1", "ch1", "tt2", "tc2", "hw2", "ch2"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Swiss route data missing columns: {missing}")

    work = df.copy()
    work["task_id"] = work.groupby("ID").cumcount() + 1
    alt_specific = {"tt1", "tc1", "hw1", "ch1", "tt2", "tc2", "hw2", "ch2"}
    covariates = [c for c in work.columns if c not in {"ID", "choice", "task_id", *alt_specific}]

    parts = []
    for alt in (1, 2):
        part = pd.DataFrame(
            {
                "respondent_id": work["ID"],
                "task_id": work["task_id"],
                "alternative": alt,
                "choice": (work["choice"] == alt).astype(int),
                "travel_time": work[f"tt{alt}"],
                "travel_cost": work[f"tc{alt}"],
                "headway": work[f"hw{alt}"],
                "changes": work[f"ch{alt}"],
            }
        )
        for cov in covariates:
            part[cov] = work[cov]
        parts.append(part)

    long_df = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["respondent_id", "task_id", "alternative"])
        .reset_index(drop=True)
    )
    return long_df


def load_apollo_long(
    dataset: str,
    raw_output_path: str | Path | None = None,
    long_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download and convert Apollo sample data to long format suitable for this toolkit.

    Supported datasets:
      - swiss_route_choice
    """
    raw = download_apollo_dataset(dataset, output_path=raw_output_path)

    if dataset == "swiss_route_choice":
        long_df = _to_long_swiss_route(raw)
    else:
        raise ValueError(f"Long-format conversion is not implemented for dataset '{dataset}'.")

    if long_output_path is not None:
        out = Path(long_output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(out, index=False)

    return long_df
