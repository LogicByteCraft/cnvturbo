# cnvturbo

[![PyPI version](https://img.shields.io/pypi/v/cnvturbo.svg)](https://pypi.org/project/cnvturbo/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)
[![Scanpy compatible](https://img.shields.io/badge/Scanpy-1.10%2B-1f77b4)](https://scanpy.readthedocs.io/)

**`cnvturbo`** — A Python re-implementation of [R inferCNV](https://github.com/broadinstitute/inferCNV) for single-cell RNA-seq copy-number variation analysis. **Algorithmically faithful to R inferCNV's HMM i6 pipeline, ~100× faster, and fully integrated with the Scanpy / AnnData ecosystem.**

> Rewritten in pure Python with R-exact algorithm alignment (hspike emission calibration, gene-level Viterbi in copy-ratio space, R-equivalent denoise + subcluster Tumor calling). The R-exact pipeline runs on CPU + joblib; optional Numba CPU / PyTorch CUDA kernels accelerate the legacy `tl.infercnv` and `tl.hmm_call_cells` paths.

---

## Why `cnvturbo`?

| Feature | R inferCNV | infercnvpy | **cnvturbo** |
|---|---|---|---|
| Cell-level Tumor/Normal HMM | ✓ | ✗ (cluster score only) | ✓ |
| HMM i6 + hspike emission | ✓ | ✗ | ✓ (analytic + MAD-robust) |
| Per-chromosome Viterbi (copy-ratio) | ✓ | ✗ | ✓ |
| Denoise (segment-length filter) | ✓ | ✗ | ✓ |
| Reference subcluster handling | ✓ | partial | ✓ |
| GPU / Numba acceleration | ✗ | ✗ | ✓ (legacy `tl.infercnv` + `tl.hmm_call_cells`; R-exact path is CPU + joblib) |
| Runtime (P12, 7,269 cells) | **~5 hr** | ~9 min | **~86 s** |
| Strict Tumor/Normal concordance with R | 1.000 (ref) | N/A (no cell-level HMM) | **F1 0.980** |

Verified on 40 PDAC samples (99,679 observation cells): **region-level CNV calls are 100% identical to R inferCNV**, strict cell-level Tumor/Normal calls reach **overall F1 = 0.980**, and per-cell continuous `cnv_score` matches R `cnv_signal_R` with mean Pearson **0.99997**. See [Benchmark](#benchmark) below.

> **Speed-up attribution**: the R-exact main pipeline (`infercnv_r_compat` +
> `compute_hspike_emission_params` + `hmm_call_subclusters`) is **CPU + joblib only**.
> All speed-up numbers in this README come from algorithmic rewrite +
> multi-core parallelism, **not** GPU. The optional GPU back-end currently
> only accelerates the legacy `tl.infercnv` (sliding-window scoring) and
> `tl.hmm_call_cells` (no-subcluster HMM) paths.

---

## Installation

### From PyPI (recommended)

```bash
pip install cnvturbo
```

### With acceleration backends

These extras are **only used by the legacy `tl.infercnv` and `tl.hmm_call_cells`
paths** (see [Backend coverage](#backend-coverage)). The R-exact main pipeline
runs on stock CPU + joblib regardless of which extra you install.

```bash
# Numba CPU kernels (legacy `tl.infercnv` sliding-window + `tl.hmm_call_cells` Viterbi)
pip install "cnvturbo[hmm-cpu]"

# PyTorch CUDA back-end (same scope as above; falls back to CPU if no GPU)
pip install "cnvturbo[hmm-gpu]"

# Everything above + Baum-Welch EM emission fitting (`hmmlearn`)
pip install "cnvturbo[hmm]"
```

### Development install

```bash
git clone https://github.com/LogicByteCraft/cnvturbo.git
cd cnvturbo
pip install -e ".[dev,test]"
```

### Requirements

* Python ≥ 3.10
* `scanpy ≥ 1.10`, `anndata ≥ 0.7.3`, `numpy ≥ 1.20`, `pandas ≥ 1`
* Optional accelerators (only effective for `tl.infercnv` + `tl.hmm_call_cells` —
  the R-exact pipeline does not use them):
    * `numba ≥ 0.57` — Numba parallel CPU kernels for sliding-window convolution
    * `torch ≥ 2.0` — PyTorch CUDA back-end for sliding-window conv1d + batched Viterbi
    * `hmmlearn ≥ 0.3` — Baum-Welch EM emission fitting (`fit_method="em"`)

---

## Quick start

```python
import scanpy as sc
import cnvturbo
from cnvturbo import tl as cnv_tl, pl as cnv_pl

adata = sc.read_h5ad("my_sample.h5ad")
adata.layers["counts"] = adata.X.copy()

cnv_tl.infercnv_r_compat(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial", "Fibroblast"],
    window_size=101,
    min_mean_expr_cutoff=0.1,    # R inferCNV default for 10x; use 1.0 for Smart-seq2
    apply_2x_transform=True,
    n_jobs=16,
)

emit_means, emit_stds, emit_sd_intercepts, emit_sd_slopes = cnv_tl.compute_hspike_emission_params(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial", "Fibroblast"],
    min_mean_expr_cutoff=0.1,    # 必须与 infercnv_r_compat 保持一致
    output_space="copy_ratio",
    return_sd_trend=True,
)

cnv_tl.hmm_call_subclusters(
    adata,
    use_rep="cnv",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial", "Fibroblast"],
    precomputed_emit_means=emit_means,
    precomputed_emit_stds=emit_stds,
    precomputed_emit_sd_intercepts=emit_sd_intercepts,
    precomputed_emit_sd_slopes=emit_sd_slopes,
    leiden_resolution="auto",
    cluster_by_groups=True,
    min_segment_length=5,
    min_segments_for_tumor=1,
    key_added="cnv_call",
    n_jobs=16,
)

print(adata.obs["cnv_call"].value_counts())
```

After this, `adata.obs["cnv_call"]` contains `"Tumor"` / `"Normal"` per cell, and `adata.obs["cnv_call_score"]` stores the HMM non-neutral state fraction (`proportion_cnv`).

For strict R-equivalent cell-level calls, combine the HMM burden with a continuous denoised CNV signal:

```python
ref_mask = adata.obs["cell_type"].isin(["NK", "Endothelial", "Fibroblast"]).to_numpy()
x_denoise = cnv_tl.denoise_r_compat(adata.obsm["X_cnv"], ref_mask)
adata.obs["cnv_score"] = np.mean(np.abs(x_denoise - 1.0), axis=1)
adata.obs["proportion_cnv"] = adata.obs["cnv_call_score"].astype(float)
adata.obs["is_obs_tumor"] = (
    (~ref_mask)
    & (adata.obs["cnv_score"] > np.percentile(adata.obs.loc[ref_mask, "cnv_score"], 95))
    & (adata.obs["proportion_cnv"] > np.percentile(adata.obs.loc[ref_mask, "proportion_cnv"], 95))
)
```

End-to-end reusable scripts are available in [`template/`](template/).

---

## Detailed usage

### 1. Prepare AnnData

`cnvturbo` requires:
* **Raw integer counts** in `adata.X` or `adata.layers["counts"]`.
* **Gene coordinates** in `adata.var`: columns `chromosome`, `start`, `end`.
* **A reference annotation** in `adata.obs`: a column identifying normal cells (e.g., NK / Endothelial / Fibroblast).

Add gene coordinates from a GTF:

```python
from cnvturbo.io import genomic_position_from_gtf

genomic_position_from_gtf(
    gtf_file="Homo_sapiens.GRCh38.110.gtf.gz",
    adata=adata,
)
```

### 2. R-compatible preprocessing (`infercnv_r_compat`)

Reproduces R inferCNV's pipeline exactly:

0. **Low-expression gene filter** — `mean(raw_count) < min_mean_expr_cutoff`
   (R `require_above_min_mean_expr_cutoff`; 10x default `0.1`, Smart-seq2 `1.0`)
1. Library-size normalization → median depth
2. `log2(x + 1)`
3. First reference subtraction (gene-space, "bounds" mode)
4. Clip to ±3 (default)
5. Per-chromosome same-length pyramid smoothing (window=101)
6. Per-cell median centering
7. Second reference subtraction (gene-space)
8. `2^x` → copy-ratio (neutral ≈ 1.0)

```python
cnv_tl.infercnv_r_compat(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial"],
    max_ref_threshold=3.0,
    window_size=101,
    exclude_chromosomes=("chrX", "chrY"),
    min_mean_expr_cutoff=0.1,    # R inferCNV default for 10x; set 1.0 for Smart-seq2; 0 to disable
    apply_2x_transform=True,
    n_jobs=16,
    key_added="cnv",
)
```

Output:
* `adata.obsm["X_cnv"]` — `(n_cells × n_genes_filtered)` copy-ratio matrix
* `adata.uns["cnv"]["chr_pos"]` — gene-level chromosome offsets
* `adata.uns["cnv"]["kept_var_names"]` — original `var_names` that survived
  `min_mean_expr_cutoff` + `chrX/chrY` exclusion (matches `obsm["X_cnv"]` columns)
* `adata.uns["cnv"]["min_mean_expr_cutoff"]` — actual cutoff applied (provenance)

### 3. hspike emission calibration (`compute_hspike_emission_params`)

Mirrors R's `hidden_spike` simulation: builds a synthetic genome (50% CNV / 50% neutral chromosomes), samples the simulation base from real reference cells, runs the full pipeline, and extracts emission parameters per CNV state.

```python
emit_means, emit_stds, emit_sd_intercepts, emit_sd_slopes = cnv_tl.compute_hspike_emission_params(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial"],
    min_mean_expr_cutoff=0.1,    # 必须与 infercnv_r_compat 保持一致
    n_sim_cells=100,
    n_genes_per_chr=400,
    output_space="copy_ratio",
    return_sd_trend=True,
)
```

### 4. HMM cell-level Tumor calling (`hmm_call_subclusters`)

R-equivalent decoder: per-group Leiden subclustering (`cluster_by_groups=True`, auto resolution), per-chromosome Viterbi with R's `pnorm`-based emission, segment-length denoise, "subcluster contains ≥1 CNV segment ⇒ Tumor" rule.

```python
cnv_tl.hmm_call_subclusters(
    adata,
    use_rep="cnv",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial"],
    precomputed_emit_means=emit_means,
    precomputed_emit_stds=emit_stds,
    precomputed_emit_sd_intercepts=emit_sd_intercepts,
    precomputed_emit_sd_slopes=emit_sd_slopes,
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
    key_added="cnv_call",
    backend="auto",
    n_jobs=16,
)
```

Output (added to `adata.obs`):
* `cnv_call` — `"Tumor"` / `"Normal"` per cell
* `cnv_call_score` — HMM non-neutral state fraction (`proportion_cnv`)
* `cnv_call_expr_deviation` — raw expression deviation (`mean(|X_cnv − 1.0|)`)
* `cnv_call_subcluster` — Leiden subcluster id used for HMM

### 5. Visualization

```python
cnv_tl.pca(adata, use_rep="cnv")
cnv_tl.umap(adata)
cnv_pl.chromosome_heatmap(adata, groupby="cnv_call")

import scanpy as sc
sc.pl.embedding(adata, basis="cnv_umap", color=["cnv_call", "cnv_call_score"])
```

---

## Benchmark

Pancreatic adenocarcinoma benchmark, 40 samples, 99,679 observation cells; reference group = NK / T-like normal cells depending on sample annotation. R inferCNV outputs were used only for validation, not as cnvturbo inputs.

| Metric | Result |
|---|---:|
| Region-level CNV call accuracy vs R | **1.000** |
| Region-level CNV call F1 vs R | **1.000** |
| Strict cell-level Tumor/Normal accuracy vs R | **0.986** |
| Strict cell-level Tumor/Normal precision vs R | **0.976** |
| Strict cell-level Tumor/Normal recall vs R | **0.984** |
| Strict cell-level Tumor/Normal F1 vs R | **0.980** |
| Per-cell `cnv_score` mean Pearson vs R `cnv_signal_R` | **0.99997** |
| Per-cell `cnv_score` max RMSE vs R `cnv_signal_R` | **1.24e-4** |

The strict call is the dual-gate rule used by the templates:
`cnv_score > P95(reference)` and `proportion_cnv > P95(reference)`.

---

## API overview

```text
cnvturbo
├── tl                              # tools
│   ├── infercnv                    # original sliding-window scoring
│   ├── infercnv_r_compat           # R-exact 8-step pipeline (recommended)
│   ├── compute_hspike_emission_params  # hspike-based HMM emission calibration
│   ├── hmm_call_subclusters        # subcluster-level R-equivalent HMM caller
│   ├── hmm_call_cells              # cell-level HMM caller (no subclustering)
│   ├── cnv_score, cnv_score_cell   # CNV burden scores
│   ├── ithcna, ithgex              # intra-tumor heterogeneity
│   ├── pca, umap, tsne, leiden     # CNV-space embeddings (Scanpy wrappers)
│   └── copykat                     # CopyKAT integration (optional, requires R)
├── pp                              # preprocessing utilities
├── pl                              # plotting
├── io                              # GTF / genomic-position helpers
└── datasets                        # bundled tutorial data
```

---

## Design highlights

* **R-exact pipeline**: `infercnv_r_compat` reproduces the full 8 R inferCNV steps in gene-space copy-ratio (vs. window-space log2 used by older Python ports).
* **HMM i6 cell-level calling**: `hmm_call_subclusters` reproduces R's HMM Viterbi decoder, denoising, and per-subcluster Tumor classification — typically absent from existing Python implementations.
* **Performance kernels**: Numba parallel CPU + PyTorch CUDA back-ends for the
  **legacy** `tl.infercnv` (sliding-window conv1d) and `tl.hmm_call_cells`
  (batched Viterbi) paths (`backend="auto" | "cpu" | "cuda"`). The R-exact path
  (`infercnv_r_compat` + `compute_hspike_emission_params` + `hmm_call_subclusters`)
  currently runs on **CPU + joblib only** — see *Backend coverage* below.
* **Robust to reference contamination**: emission std uses MAD (median absolute deviation) × 1.4826 instead of plain std, so reference cells contaminated by tumor cells don't inflate state widths.

A high-level `infercnv` / `cnv_score` / `chromosome_heatmap` API similar to the de facto Python convention is also exposed for ease of migration.

### Backend coverage

| Function | Numba CPU | PyTorch CUDA | Notes |
|---|---|---|---|
| `tl.infercnv` (legacy sliding-window scoring) | ✓ | ✓ | `backend="auto"` picks GPU when available |
| `tl.hmm_call_cells` (cell-level HMM, no subcluster) | ✓ | ✓ | same |
| `tl.infercnv_r_compat` (**R-exact 8-step pipeline**) | — | — | CPU + `joblib` (`n_jobs`); no GPU code path |
| `tl.compute_hspike_emission_params` | — | — | same |
| `tl.hmm_call_subclusters` (**R-exact subcluster HMM**) | — | — | `use_r_viterbi=True` (default) is hard-wired to the R-pnorm CPU Viterbi; `backend` argument is currently a no-op on this path |

**Practical implication.** If you follow the recommended `infercnv_r_compat`
+ `hmm_call_subclusters` workflow, install `cnvturbo` without any accelerator
extra and tune `n_jobs` / `OMP_NUM_THREADS` for CPU throughput. GPU extras
only help if you use the legacy `tl.infercnv` / `tl.hmm_call_cells` paths.
Wiring the R-exact subcluster Viterbi onto GPU is on the roadmap.

---

## Citation

If you use `cnvturbo` in your research, please cite this implementation:

```bibtex
@software{cnvturbo,
  title  = {cnvturbo: A high-performance scRNA-seq CNV inference toolkit with R inferCNV-compatible HMM i6 (CPU + optional GPU back-ends)},
  url    = {https://github.com/LogicByteCraft/cnvturbo},
  year   = {2026}
}
```

`cnvturbo`'s algorithm is a faithful port of [R inferCNV](https://github.com/broadinstitute/inferCNV); please cite the upstream methodology as well when relevant.

---

## License

BSD 3-Clause License — see [`LICENSE`](LICENSE).

## Acknowledgements

`cnvturbo` is inspired by and stays algorithmically aligned with:

* [`inferCNV`](https://github.com/broadinstitute/inferCNV) — reference R implementation of the HMM i6 pipeline.
* [`Scanpy`](https://scanpy.readthedocs.io/) / [`AnnData`](https://anndata.readthedocs.io/) — single-cell analysis ecosystem.

---

## Contributing

Issues and pull requests are welcome at <https://github.com/LogicByteCraft/cnvturbo>. Before contributing:

```bash
pip install -e ".[dev,test]"
pre-commit install
pytest
```
