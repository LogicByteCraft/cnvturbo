# cnvturbo

[![PyPI version](https://img.shields.io/pypi/v/cnvturbo.svg)](https://pypi.org/project/cnvturbo/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)
[![Scanpy compatible](https://img.shields.io/badge/Scanpy-1.10%2B-1f77b4)](https://scanpy.readthedocs.io/)

**`cnvturbo`** — A Python re-implementation of [R inferCNV](https://github.com/broadinstitute/inferCNV) for single-cell RNA-seq copy-number variation analysis. **Algorithmically faithful to R inferCNV's HMM i6 pipeline, ~100× faster, and fully integrated with the Scanpy / AnnData ecosystem.**

> Rewritten in pure Python with R-exact algorithm alignment (hspike emission calibration, gene-level Viterbi in copy-ratio space, R-equivalent denoise + subcluster Tumor calling), plus Numba/CUDA-accelerated kernels.

---

## Why `cnvturbo`?

| Feature | R inferCNV | infercnvpy | **cnvturbo** |
|---|---|---|---|
| Cell-level Tumor/Normal HMM | ✓ | ✗ (cluster score only) | ✓ |
| HMM i6 + hspike emission | ✓ | ✗ | ✓ (analytic + MAD-robust) |
| Per-chromosome Viterbi (copy-ratio) | ✓ | ✗ | ✓ |
| Denoise (segment-length filter) | ✓ | ✗ | ✓ |
| Reference subcluster handling | ✓ | partial | ✓ |
| GPU / Numba acceleration | ✗ | ✗ | ✓ |
| Runtime (P12, 7,269 cells) | **~5 hr** | ~9 min | **~86 s** |
| Cell-level concordance with R | 1.000 (ref) | 0.81 | **1.000** |

Verified on 3 PDAC samples (15,135 cells total): **cell-level Tumor/Normal classification 100% identical to R inferCNV's HMM output**, while running 100–200× faster. See [Benchmark](#benchmark) below.

---

## Installation

### From PyPI (recommended)

```bash
pip install cnvturbo
```

### With acceleration backends

```bash
# CPU acceleration (Numba)
pip install "cnvturbo[hmm-cpu]"

# GPU acceleration (PyTorch)
pip install "cnvturbo[hmm-gpu]"

# All accelerators + EM fitting
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
* Optional: `numba ≥ 0.57` (CPU), `torch ≥ 2.0` (GPU), `hmmlearn ≥ 0.3` (EM)

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

emit_means, emit_stds = cnv_tl.compute_hspike_emission_params(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial", "Fibroblast"],
    min_mean_expr_cutoff=0.1,    # 必须与 infercnv_r_compat 保持一致
    output_space="copy_ratio",
)

cnv_tl.hmm_call_subclusters(
    adata,
    use_rep="cnv",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial", "Fibroblast"],
    precomputed_emit_means=emit_means,
    precomputed_emit_stds=emit_stds,
    leiden_resolution="auto",
    cluster_by_groups=True,
    min_segment_length=5,
    min_segments_for_tumor=1,
    key_added="cnv_call",
    n_jobs=16,
)

print(adata.obs["cnv_call"].value_counts())
```

After this, `adata.obs["cnv_call"]` contains `"Tumor"` / `"Normal"` per cell, and `adata.obs["cnv_call_score"]` carries a continuous CNV burden score (`mean(|X_cnv − 1.0|)` in copy-ratio space).

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
emit_means, emit_stds = cnv_tl.compute_hspike_emission_params(
    adata,
    raw_layer="counts",
    reference_key="cell_type",
    reference_cat=["NK", "Endothelial"],
    min_mean_expr_cutoff=0.1,    # 必须与 infercnv_r_compat 保持一致
    n_sim_cells=100,
    n_genes_per_chr=400,
    output_space="copy_ratio",
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
    leiden_resolution="auto",
    cluster_by_groups=True,
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
* `cnv_call_score` — continuous CNV burden (`mean(|X_cnv − 1.0|)`)
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

Three pancreatic adenocarcinoma samples (P07 = 3,659 cells, P12 = 7,269 cells, P30 = 4,207 cells); reference group = NK + Endothelial + Fibroblast (~50% of all cells).

| Sample | R inferCNV (runtime) | cnvturbo (runtime) | Speed-up | cnvturbo cell-level Accuracy vs R |
|---|---|---|---|---|
| P07CRX_T (3,659) | 2.5 h | 64 s | **140×** | **1.000** |
| P12HWZ_T (7,269) | 5.0 h | 86 s | **210×** | **1.000** |
| P30WJJ_T (4,207) | 3.5 h | 54 s | **230×** | **1.000** |

cnvturbo's per-cell `Tumor` / `Normal` classification is **identical** to R inferCNV's HMM output across all 15,135 cells.

> The "ground truth" was reconstructed directly from R's `pred_cnv_regions.dat` + `cell_groupings` to bypass a known fuzzy-match bug in some user post-processing scripts.

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
* **Performance kernels**: Numba parallel CPU / PyTorch GPU back-ends for sliding-window convolution and batched Viterbi (`backend="auto" | "cpu" | "cuda"`).
* **Robust to reference contamination**: emission std uses MAD (median absolute deviation) × 1.4826 instead of plain std, so reference cells contaminated by tumor cells don't inflate state widths.

A high-level `infercnv` / `cnv_score` / `chromosome_heatmap` API similar to the de facto Python convention is also exposed for ease of migration.

---

## Citation

If you use `cnvturbo` in your research, please cite this implementation:

```bibtex
@software{cnvturbo,
  title  = {cnvturbo: GPU/Numba-accelerated scRNA-seq CNV inference with R inferCNV-compatible HMM i6},
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
