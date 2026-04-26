# ── 0. Usage ────────────────────────────────────────────────────────────────
# Optional script for users who already exported R inferCNV per-cell calls.
#
# Expected R CSV columns:
#   cell, cnv_signal_R, proportion_cnv_R, tumor_strict_R
#
# Run:
#   cd /path/to/project
#   python cnvturbo/template/03_benchmark_against_infercnv.py
# ────────────────────────────────────────────────────────────────────────────


# ── 1. Standard library imports ─────────────────────────────────────────────
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── 2. Third-party imports ─────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ── 3. Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cnvturbo_template_03")


# ── 4. Configuration ───────────────────────────────────────────────────────
CNVTURBO_OBS_DIR = Path("output/cnvturbo_template")
R_CALLS_DIR = Path("output/r_infercnv_calls")
OUTPUT_CSV = CNVTURBO_OBS_DIR / "benchmark_vs_r_infercnv.csv"


# ── 5. Metrics ─────────────────────────────────────────────────────────────
def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    n = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else np.nan

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _score_metrics(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    xv = x.astype(float).to_numpy()
    yv = y.astype(float).to_numpy()
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) == 0:
        return {"n_score": 0, "pearson": np.nan, "spearman": np.nan, "rmse": np.nan}
    return {
        "n_score": int(len(xv)),
        "pearson": float(pd.Series(xv).corr(pd.Series(yv), method="pearson")),
        "spearman": float(pd.Series(xv).corr(pd.Series(yv), method="spearman")),
        "rmse": float(np.sqrt(np.mean((xv - yv) ** 2))),
    }


# ── 6. I/O ─────────────────────────────────────────────────────────────────
def _load_cnvturbo_obs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index(names="cell")
    if "is_obs_tumor" not in df.columns:
        raise KeyError(f"{path} does not contain is_obs_tumor. Run script 01 first.")
    return df


def _load_r_calls(sample: str) -> pd.DataFrame:
    path = R_CALLS_DIR / f"{sample}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing R inferCNV calls: {path}")
    df = pd.read_csv(path)
    required = {"cell", "tumor_strict_R"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    return df


# ── 7. Main benchmark ──────────────────────────────────────────────────────
def main() -> None:
    rows = []
    obs_files = sorted(CNVTURBO_OBS_DIR.glob("*.cnvturbo_obs.csv"))
    if not obs_files:
        raise FileNotFoundError(f"No *.cnvturbo_obs.csv files found in {CNVTURBO_OBS_DIR}")

    for obs_path in obs_files:
        sample = obs_path.name.replace(".cnvturbo_obs.csv", "")
        ct = _load_cnvturbo_obs(obs_path)
        r = _load_r_calls(sample)
        merged = ct.merge(r, on="cell", how="inner", validate="one_to_one")
        if merged.empty:
            raise ValueError(f"{sample}: no overlapping cell barcodes between cnvturbo and R CSV")

        y_true = merged["tumor_strict_R"].astype(str).str.lower().eq("tumor").to_numpy()
        y_pred = merged["is_obs_tumor"].astype(bool).to_numpy()

        row = {"sample": sample}
        row.update(_binary_metrics(y_true, y_pred))
        if {"cnv_signal_R", "cnv_score"}.issubset(merged.columns):
            row.update(_score_metrics(merged["cnv_signal_R"], merged["cnv_score"]))
        rows.append(row)
        logger.info("%s: accuracy=%.4f F1=%.4f", sample, row["accuracy"], row["f1"])

    out = pd.DataFrame(rows).sort_values("sample")
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved benchmark table: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
