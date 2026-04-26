# ── 0. Usage ────────────────────────────────────────────────────────────────
# Recommended environment:
#   mamba create -n cnvturbo_env python=3.10 -y
#   mamba activate cnvturbo_env
#   pip install cnvturbo scanpy anndata pandas numpy matplotlib seaborn
#
# Run after 01_r_compatible_analysis.py:
#   cd /path/to/project
#   python cnvturbo/template/02_visualization.py
# ────────────────────────────────────────────────────────────────────────────


# ── 1. Standard library imports ─────────────────────────────────────────────
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── 2. Third-party imports ─────────────────────────────────────────────────
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


# ── 3. Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cnvturbo_template_02")


# ── 4. Reproducibility ─────────────────────────────────────────────────────
SEED = 1234
np.random.seed(SEED)
sc.settings.set_figure_params(dpi=120, dpi_save=300, frameon=False, fontsize=10)


# ── 5. Configuration ───────────────────────────────────────────────────────
INPUT_DIR = Path("output/cnvturbo_template")
OUTPUT_DIR = INPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H5AD_FILES = sorted(INPUT_DIR.glob("*.cnvturbo.h5ad"))
UMAP_PARAMS = dict(n_pcs=30, n_neighbors=20, random_state=SEED)


# ── 6. Helpers ─────────────────────────────────────────────────────────────
def _load_data() -> ad.AnnData:
    if not H5AD_FILES:
        raise FileNotFoundError(f"No *.cnvturbo.h5ad files found in {INPUT_DIR}")

    adatas = []
    for path in H5AD_FILES:
        sample = path.name.replace(".cnvturbo.h5ad", "")
        one = sc.read_h5ad(path)
        if "sample" not in one.obs:
            one.obs["sample"] = sample
        one.obs["sample"] = one.obs["sample"].astype(str)
        adatas.append(one)
        logger.info("Loaded %s: %d cells x %d genes", path, one.n_obs, one.n_vars)

    if len(adatas) == 1:
        return adatas[0]
    return ad.concat(adatas, join="outer", label="template_batch", keys=[p.stem for p in H5AD_FILES])


def _derive_cnv_state(adata: ad.AnnData) -> None:
    if "is_obs_tumor" not in adata.obs:
        raise KeyError("adata.obs['is_obs_tumor'] is required. Run script 01 first.")

    is_tumor = adata.obs["is_obs_tumor"].astype(bool).to_numpy()
    is_normal = adata.obs.get("is_obs_normal", pd.Series(False, index=adata.obs_names)).astype(bool).to_numpy()
    state = np.where(is_tumor, "Observation_Tumor", np.where(is_normal, "Observation_Normal", "Reference"))
    adata.obs["cnv_state"] = pd.Categorical(
        state,
        categories=["Reference", "Observation_Normal", "Observation_Tumor"],
        ordered=True,
    )


def _ensure_umap(adata: ad.AnnData) -> None:
    if "X_umap" in adata.obsm:
        return
    sc.pp.pca(adata, n_comps=UMAP_PARAMS["n_pcs"], random_state=SEED)
    sc.pp.neighbors(adata, n_neighbors=UMAP_PARAMS["n_neighbors"], n_pcs=UMAP_PARAMS["n_pcs"])
    sc.tl.umap(adata, random_state=UMAP_PARAMS["random_state"])


def _plot_umap(adata: ad.AnnData) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc.pl.umap(adata, color="sample", ax=axes[0], show=False, frameon=False, title="Sample")
    sc.pl.umap(adata, color="cnv_state", ax=axes[1], show=False, frameon=False, title="CNV state")
    sc.pl.umap(
        adata,
        color="cnv_score",
        ax=axes[2],
        show=False,
        frameon=False,
        color_map="magma",
        title="CNV score",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "umap_cnv_overview.png", dpi=300)
    plt.close(fig)


def _plot_tumor_fraction(adata: ad.AnnData) -> None:
    df = (
        adata.obs.assign(is_obs_tumor=adata.obs["is_obs_tumor"].astype(bool))
        .groupby("sample", observed=True)["is_obs_tumor"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.35), 4))
    df.plot.bar(ax=ax, color="#D62728")
    ax.set_ylabel("Observation tumor cells (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tumor_fraction_per_sample.png", dpi=300)
    plt.close(fig)


def _plot_score_density(adata: ad.AnnData) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for state, color in [
        ("Reference", "#90A4AE"),
        ("Observation_Normal", "#1F77B4"),
        ("Observation_Tumor", "#D62728"),
    ]:
        values = adata.obs.loc[adata.obs["cnv_state"] == state, "cnv_score"].astype(float)
        if len(values) > 0:
            values.plot(kind="kde", ax=ax, label=state, color=color)
    ax.set_xlabel("cnv_score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cnv_score_density.png", dpi=300)
    plt.close(fig)


# ── 7. Main visualization ──────────────────────────────────────────────────
def main() -> None:
    adata = _load_data()
    _derive_cnv_state(adata)
    _ensure_umap(adata)

    _plot_umap(adata)
    _plot_tumor_fraction(adata)
    _plot_score_density(adata)
    logger.info("Saved figures to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
