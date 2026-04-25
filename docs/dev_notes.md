# Developer Notes / 开发者注意事项

This page collects **non-obvious lessons** that downstream users keep
re-discovering. If you are writing analysis scripts on top of
``cnvturbo`` (especially R-compatible workflows), please skim this page
before rolling your own helpers.

---

## 1. Don't rebuild cluster-level Viterbi in your own scripts

> **TL;DR** — Use ``adata.obs["{key_added}_score"]`` (written by
> {func}`cnvturbo.tl.hmm_call_subclusters`) as the per-cell
> *proportion-of-non-neutral-windows* metric.
> **Do not** re-run cluster-mean Viterbi from outside the library.

### What looked tempting

R `inferCNV` exposes a per-cell metric `proportion_cnv_R = mean(state ≠ neutral)`
on the `17_HMM` matrix. A natural Python equivalent is to:

1. take the cluster mean of `X_cnv_r` per leiden subcluster,
2. feed it back into the same R-style Viterbi used internally,
3. broadcast the per-cluster state path back to every cell.

That is exactly what `hmm_call_subclusters` does internally, and it is
*also* what several external scripts in the wild end up doing
(e.g. an early ``_rebuild_proportion_cnv`` helper). On well-behaved
samples (many small subclusters of comparable size) the external
rebuild matches the internal one bit-for-bit.

### Why it silently breaks

The internal cluster Viterbi shrinks the emission std by cluster size:

```python
sd_ci = base_sd / sqrt(n_cells_in_cluster)
```

This makes mathematical sense: the cluster mean is an average of `n`
draws and its sampling noise scales as `1/sqrt(n)`. But the formula
has two failure modes that an *external* rebuild rarely accounts for:

* **Pathologically large reference clusters.** When `cluster_by_groups=True`
  and the reference group collapses into a single dominant subcluster
  (e.g. a sample with one large `Reference_0` of >3,000 cells),
  `sd_ci` collapses to near zero. The Gaussian emission becomes a
  delta-like spike, and even cluster means within `0.001` of `1.0`
  get pulled into a non-neutral state by the *transition* term. The
  result: reference cells end up with `proportion_cnv ≈ 1.0`, the
  P95 reference threshold becomes `≈ 1.0`, and every observation cell
  silently fails the "strict double P95" gate.

* **Parameter drift between caller and library.** The internal Viterbi
  consumes `emit_means`, `emit_stds`, `p_stay`, `min_segment_length`,
  `_R_LOG_PRIOR`, `chr_pos`, `cluster_by_groups`, leiden resolution,
  `min_mean_expr_cutoff`, `exclude_chromosomes`. *Any* of these drifting
  out of sync between an external rebuild and the next library release
  produces results that look plausible but are wrong by a quiet margin.

We hit both at once in the `cnvturbo>=0.2.0` PDAC pipeline: enabling
`min_mean_expr_cutoff=0.1` reshuffled the gene set, which in turn
collapsed the reference into one giant subcluster, and the external
rebuild produced `Reference_P95 ≈ 1.0`.

### What you should do instead

`hmm_call_subclusters` already writes the cell-level *non-neutral
window fraction* into `adata.obs[f"{key_added}_score"]` (default
`adata.obs["cnv_tumor_call_score"]`). That column **is** the
algorithmic equivalent of R `proportion_cnv_R`, computed by exactly
the same Viterbi instance the library trusts.

```python
import cnvturbo as cnv

cnv.tl.infercnv_r_compat(adata, ..., key_added="cnv_r")
em, es = cnv.tl.compute_hspike_emission_params(adata, ...)
cnv.tl.hmm_call_subclusters(
    adata,
    use_rep="cnv_r",
    precomputed_emit_means=em,
    precomputed_emit_stds=es,
    key_added="ct_call",
)

# Use this directly — no manual Viterbi rebuild needed.
proportion_cnv = adata.obs["ct_call_score"].to_numpy()

# Per-cell continuous CNV signal (R cnv_signal_R equivalent):
import numpy as np, scipy.sparse as sp
X = adata.obsm["X_cnv_r"]
X = X.toarray() if sp.issparse(X) else np.asarray(X)
cnv_signal = np.abs(X - 1.0).mean(axis=1)

# "Top-tier strict" tumor call (P95 of reference, two-metric AND gate):
is_ref = (adata.obs["infercnv_group"] == "Reference_Normal").to_numpy()
th_signal = np.percentile(cnv_signal[is_ref], 95)
th_prop   = np.percentile(proportion_cnv[is_ref], 95)
is_obs    = ~is_ref
adata.obs["is_obs_tumor"] = is_obs & (cnv_signal > th_signal) & (proportion_cnv > th_prop)
```

### Cross-check (PDAC 3-sample benchmark, v0.2.1)

`obs["ct_call_score"]` and a (carefully matched) external
`reconstruct_cluster_state_matrix` produce *identical* downstream
strict tumor calls on healthy inputs:

| Sample      | strict acc | F1     | Cohen κ | TP/TN/FP/FN          |
|-------------|-----------:|-------:|--------:|----------------------|
| P07CRX_T    |     0.802  | 0.745  |   0.595 | 57 / 101 / 36 / 3    |
| P12HWZ_T    |     0.945  | 0.849  |   0.816 | 578 / 2975 / 148 / 57|
| P30WJJ_T    |     0.967  | 0.933  |   0.911 | 537 / 1728 / 69 / 8  |

But once any input drifts (different `min_mean_expr_cutoff`, different
leiden resolution, different cluster-size distribution), only
`obs["ct_call_score"]` stays correct.

### Bottom line for downstream authors

* **Region-level tumor call** (R "subcluster contains ≥1 CNV segment"):
  use `adata.obs["{key_added}"]` directly. This is already the R
  `pred_cnv_regions.dat` equivalent.
* **Per-cell proportion-style metric**: use
  `adata.obs["{key_added}_score"]`. Don't rebuild it.
* **Per-cell continuous CNV signal** (R `cnv_signal_R`): compute
  `mean(|X_cnv_r - 1|)` directly on `adata.obsm["X_cnv_r"]`. Matrix-only,
  no HMM needed, bit-stable across cnvturbo versions.

---

## 2. Per-cell scoring helper: `cnv_score_cell` caveats

`cnvturbo.tl.cnv_score_cell` has two execution paths:

1. **HMM-based path** — requires
   `adata.obsm[f"X_{key_added}_states"]` to be populated, which is
   only written by {func}`cnvturbo.tl.hmm_call_cells`, **not** by
   {func}`cnvturbo.tl.hmm_call_subclusters`. The function defaults to
   `neutral_state=2` for its own state-numbering convention; if you
   want R i6 alignment you must pass `neutral_state=3`.
2. **Fallback path** — `mean(|X_cnv|)`, **without** subtracting the
   neutral baseline `1.0`. Numerically the fallback path is offset by
   ~1.0 from the R-style `mean(|X_cnv - 1|)` and is *not* directly
   comparable to R `cnv_signal_R`.

For most R-aligned workflows, prefer computing
`mean(|X_cnv_r - 1|)` directly on `adata.obsm["X_cnv_r"]` — it is the
exact algebraic counterpart of R inferCNV's `cnv_signal_R` and avoids
both pitfalls above.

---

## 3. GTF parsing on Categorical chromosome columns (fixed in 0.2.1)

`gtfparse>=2.0` returns the `seqname` column as a `pandas.Categorical`
(or `ArrowExtensionArray`) backed by `pyarrow`. Until v0.2.1,
{func}`cnvturbo.io.genomic_position_from_gtf` raised
`TypeError: can only concatenate str (not "Categorical") to str`
when prepending the `"chr"` prefix.

If you must support pre-0.2.1 environments and cannot upgrade, cast
the relevant columns to `str` *before* handing the dataframe to any
downstream code:

```python
gtf["seqname"]   = gtf["seqname"].astype(str)
gtf["gene_id"]   = gtf["gene_id"].astype(str)
gtf["gene_name"] = gtf["gene_name"].astype(str)
```

Otherwise, simply upgrade: `pip install -U "cnvturbo>=0.2.1"`.
A regression test
(`tests/test_io.py::test_get_genomic_position_from_gtf_categorical_chromosome`)
guards against future regressions.

---

## 4. Choosing parameters that match R `inferCNV` defaults

`infercnv_r_compat` defaults are tuned to match R `inferCNV` under
`HMM_type='i6'` + `analysis_mode='subclusters'` + `denoise=TRUE`.
The most common drift sources:

| Knob                       | R default | cnvturbo default  | Notes                                       |
|----------------------------|-----------|-------------------|---------------------------------------------|
| `window_size`              | 101       | 101               | odd, gene-level                             |
| `max_ref_threshold`        | 3.0       | 3.0               | bounds clip after ref subtract              |
| `min_mean_expr_cutoff`     | 0.1 (10x) | 0.1               | R `require_above_min_mean_expr_cutoff`; **set to 1.0 for Smart-seq2** |
| `exclude_chromosomes`      | chrX/chrY | `("chrX","chrY")` | pass `None` to keep them                    |
| `apply_2x_transform`       | TRUE      | True              | needed for HMM `i6` (copy-ratio space)      |
| HMM `p_stay`               | 0.99      | 0.99              | sticky transition                           |
| HMM `min_segment_length`   | 5         | 5                 | denoise-equivalent                          |
| HMM `cluster_by_groups`    | TRUE      | True              | reference and observation cluster separately|
| HMM `leiden_resolution`    | auto      | "auto"            | R: `(11.98 / n_cells) ^ (1/1.165)`          |

If you reproduce R behavior bit-for-bit, **do not change these without
regression testing against R outputs**.
