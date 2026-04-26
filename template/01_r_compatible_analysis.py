# ── 0. Usage ────────────────────────────────────────────────────────────────
# Recommended environment:
#   mamba create -n cnvturbo_env python=3.10 -y
#   mamba activate cnvturbo_env
#   pip install "cnvturbo[gtf]" scanpy anndata pandas numpy scipy matplotlib
#
# Run:
#   cd /path/to/project
#   python cnvturbo/template/01_r_compatible_analysis.py
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
import scanpy as sc

from cnvturbo import tl as cnv_tl
from cnvturbo.io import genomic_position_from_gtf


# ── 3. Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cnvturbo_template_01")


# ── 4. Reproducibility ─────────────────────────────────────────────────────
SEED = 1234
np.random.seed(SEED)


# ── 5. Configuration ───────────────────────────────────────────────────────
INPUT_H5AD = Path("data/my_sample.h5ad")
OUTPUT_DIR = Path("output/cnvturbo_template")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_NAME = INPUT_H5AD.stem
OUTPUT_H5AD = OUTPUT_DIR / f"{SAMPLE_NAME}.cnvturbo.h5ad"
OUTPUT_OBS_CSV = OUTPUT_DIR / f"{SAMPLE_NAME}.cnvturbo_obs.csv"

# Set to None if adata.var already has chromosome/start/end columns.
GTF_FILE: Path | None = Path("reference/Homo_sapiens.GRCh38.gtf.gz")

RAW_LAYER = "counts"
REFERENCE_KEY = "cell_type"
REFERENCE_CATEGORIES = ["NK", "Endothelial", "Fibroblast"]
N_JOBS = 16

INFERCNV_PARAMS = dict(
    max_ref_threshold=3.0,
    window_size=101,
    exclude_chromosomes=("chrM", "chrMT", "MT", "M"),
    min_mean_expr_cutoff=0.1,
    apply_2x_transform=True,
)

HSPIKE_PARAMS = dict(
    n_sim_cells=100,
    n_genes_per_chr=400,
    output_space="copy_ratio",
    random_state=SEED,
    return_sd_trend=True,
)

HMM_PARAMS = dict(
    leiden_resolution="auto",
    cluster_by_groups=True,
    z_score_filter=0.8,
    leiden_function="CPM",
    leiden_graph_method="seurat_snn",
    n_neighbors=20,
    n_pcs=10,
    min_segment_length=5,
    min_segments_for_tumor=1,
    use_r_viterbi=True,
    random_state=SEED,
    key_added="cnv_call",
)


# ── 6. Helpers ─────────────────────────────────────────────────────────────
def _ensure_counts_layer(adata: sc.AnnData) -> None:
    if RAW_LAYER not in adata.layers:
        logger.warning("adata.layers[%r] not found; using adata.X as raw counts.", RAW_LAYER)
        adata.layers[RAW_LAYER] = adata.X.copy()


def _reference_mask(adata: sc.AnnData) -> np.ndarray:
    ref_values = [str(x) for x in REFERENCE_CATEGORIES]
    return adata.obs[REFERENCE_KEY].astype(str).isin(ref_values).to_numpy()


def _add_strict_calls(adata: sc.AnnData) -> None:
    ref_mask = _reference_mask(adata)
    if not ref_mask.any():
        raise ValueError("No reference cells found. Check REFERENCE_KEY and REFERENCE_CATEGORIES.")

    x_denoise = cnv_tl.denoise_r_compat(adata.obsm["X_cnv"], ref_mask)
    cnv_score = np.mean(np.abs(x_denoise - 1.0), axis=1)
    proportion_cnv = adata.obs["cnv_call_score"].astype(float).to_numpy()

    signal_p95 = float(np.percentile(cnv_score[ref_mask], 95))
    prop_p95 = float(np.percentile(proportion_cnv[ref_mask], 95))

    pass_signal = cnv_score > signal_p95
    pass_prop = proportion_cnv > prop_p95
    is_obs_tumor = (~ref_mask) & pass_signal & pass_prop

    adata.obsm["X_cnv_denoise"] = x_denoise
    adata.obs["cnv_score"] = cnv_score
    adata.obs["proportion_cnv"] = proportion_cnv
    adata.obs["pass_signal_p95"] = pass_signal
    adata.obs["pass_prop_p95"] = pass_prop
    adata.obs["is_obs_tumor"] = is_obs_tumor
    adata.obs["is_obs_normal"] = (~ref_mask) & (~is_obs_tumor)
    adata.uns["cnvturbo_strict_thresholds"] = {
        "cnv_score_ref_p95": signal_p95,
        "proportion_cnv_ref_p95": prop_p95,
    }

    logger.info(
        "Strict tumor calls: %d/%d observation cells",
        int(is_obs_tumor.sum()),
        int((~ref_mask).sum()),
    )


# ── 7. Main analysis ───────────────────────────────────────────────────────
def main() -> None:
    adata = sc.read_h5ad(INPUT_H5AD)
    logger.info("Loaded %s: %d cells x %d genes", INPUT_H5AD, adata.n_obs, adata.n_vars)

    _ensure_counts_layer(adata)

    if GTF_FILE is not None:
        genomic_position_from_gtf(
            gtf_file=GTF_FILE,
            adata=adata,
            gtf_gene_id="auto",
            inplace=True,
        )

    cnv_tl.infercnv_r_compat(
        adata,
        raw_layer=RAW_LAYER,
        reference_key=REFERENCE_KEY,
        reference_cat=REFERENCE_CATEGORIES,
        n_jobs=N_JOBS,
        key_added="cnv",
        **INFERCNV_PARAMS,
    )

    emit_means, emit_stds, emit_sd_intercepts, emit_sd_slopes = (
        cnv_tl.compute_hspike_emission_params(
            adata,
            raw_layer=RAW_LAYER,
            reference_key=REFERENCE_KEY,
            reference_cat=REFERENCE_CATEGORIES,
            n_jobs=N_JOBS,
            min_mean_expr_cutoff=INFERCNV_PARAMS["min_mean_expr_cutoff"],
            exclude_chromosomes=INFERCNV_PARAMS["exclude_chromosomes"],
            window_size=INFERCNV_PARAMS["window_size"],
            max_ref_threshold=INFERCNV_PARAMS["max_ref_threshold"],
            **HSPIKE_PARAMS,
        )
    )

    cnv_tl.hmm_call_subclusters(
        adata,
        use_rep="cnv",
        reference_key=REFERENCE_KEY,
        reference_cat=REFERENCE_CATEGORIES,
        precomputed_emit_means=emit_means,
        precomputed_emit_stds=emit_stds,
        precomputed_emit_sd_intercepts=emit_sd_intercepts,
        precomputed_emit_sd_slopes=emit_sd_slopes,
        n_jobs=N_JOBS,
        **HMM_PARAMS,
    )

    _add_strict_calls(adata)

    adata.write_h5ad(OUTPUT_H5AD)
    adata.obs.to_csv(OUTPUT_OBS_CSV)
    logger.info("Saved AnnData: %s", OUTPUT_H5AD)
    logger.info("Saved obs table: %s", OUTPUT_OBS_CSV)

    summary = pd.Series(
        {
            "sample": SAMPLE_NAME,
            "n_cells": adata.n_obs,
            "n_genes_after_filter": adata.obsm["X_cnv"].shape[1],
            "n_reference": int(_reference_mask(adata).sum()),
            "n_observation_tumor": int(adata.obs["is_obs_tumor"].sum()),
        }
    )
    logger.info("Summary:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
