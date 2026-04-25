.. _infercnv-method:

The R-compatible inferCNV pipeline
==================================

``cnvturbo`` provides a Python implementation of R
`inferCNV <https://github.com/broadinstitute/inferCNV/>`_ that is
**bit-for-bit aligned** with the reference R workflow under
``HMM_type='i6'`` + ``analysis_mode='subclusters'`` + ``denoise=TRUE``.
It is benchmarked on >40 PDAC samples with 100% subcluster-level
agreement on the R ``pred_cnv_regions.dat`` output, while running
100–230× faster.

The full pipeline consists of three stages, each exposed as a
top-level function:

1. :func:`cnvturbo.tl.infercnv_r_compat` — the R-compatible
   8-step CNV-matrix preprocessing
2. :func:`cnvturbo.tl.compute_hspike_emission_params` — hspike
   simulation to calibrate the i6 HMM emission parameters
3. :func:`cnvturbo.tl.hmm_call_subclusters` — leiden subclustering
   + cluster-level R-style Viterbi + ``denoise=TRUE`` segment-length
   filtering

.. _input-data:

Preparing input data
--------------------

Required ``AnnData`` layout:

* ``adata.layers["counts"]`` — raw integer counts (preferred), or
  raw counts in ``adata.X``.
* ``adata.var`` — must contain ``chromosome``, ``start``, ``end``
  columns. Use :func:`cnvturbo.io.genomic_position_from_gtf` or
  :func:`cnvturbo.io.genomic_position_from_biomart` to populate them.
* ``adata.obs[reference_key]`` — a column flagging known-normal cells
  (e.g. immune / stromal references) for the ``reference_cat`` value(s).

Stage 1 — R-compatible CNV preprocessing
----------------------------------------

:func:`cnvturbo.tl.infercnv_r_compat` reproduces the R inferCNV
8-step preprocessing on a per-gene basis (no window down-sampling).
The function default settings match the R ``infercnv::run`` defaults:

1. **Library-size normalization** — divide each cell by its total count
   and multiply by the median library depth.
2. **log2(x + 1)** — log transform.
3. **First bounds-mode reference subtraction** (gene-level) — for each
   gene, subtract the per-reference-category mean; values that fall
   between the min and max of the reference means are unchanged
   (avoids false positives from cell-type-specific gene clusters such
   as Ig / HLA).
4. **Clip** to ``±max_ref_threshold``.
5. **Same-length per-chromosome sliding-window smoothing**
   (``window_size=101``).
6. **Per-cell median centering** (after smoothing).
7. **Second bounds-mode reference subtraction** (still gene-level).
8. **2^x** (when ``apply_2x_transform=True``) — moves into the
   copy-ratio space (neutral ≈ 1.0) expected by the i6 HMM.

Output:

* ``adata.obsm[f"X_{key_added}"]`` — the gene-level CNV matrix
  (``cells × kept_genes``, dtype float32).
* ``adata.uns[key_added]`` — ``{"chr_pos": ..., "kept_var_names": ...,
  "min_mean_expr_cutoff": ...}``.

The optional ``min_mean_expr_cutoff=0.1`` (R
``require_above_min_mean_expr_cutoff``) drops genes whose mean raw
count is below the cutoff. **Set to 1.0 for Smart-seq2** full-length
data; pass ``0`` to disable (legacy v0.1.0 behavior).

Stage 2 — hspike emission calibration
-------------------------------------

:func:`cnvturbo.tl.compute_hspike_emission_params` simulates a
"hospital-grade spike-in" from the reference-cell distribution to
calibrate the per-state emission means and stds of the i6 HMM. This
is the same procedure as R ``inferCNV`` ``estimate_hspike_emission``,
but vectorized and parallelized.

It is normally called once per sample and the resulting
``(emit_means, emit_stds)`` arrays are passed into
:func:`cnvturbo.tl.hmm_call_subclusters` via
``precomputed_emit_means`` / ``precomputed_emit_stds`` to ensure both
stages share the exact same emission distribution.

Stage 3 — subcluster HMM with R-equivalent Viterbi
--------------------------------------------------

:func:`cnvturbo.tl.hmm_call_subclusters` reproduces R inferCNV's
``analysis_mode='subclusters'`` step-by-step (see
``inferCNV/R/inferCNV_HMM.R`` and ``inferCNV_tumor_subclusters.R``):

1. **Leiden subclustering** — when ``cluster_by_groups=True`` the
   reference and observation groups cluster independently. Default
   resolution is ``"auto"`` ⇒ ``(11.98 / n_cells)^(1/1.165)`` (R
   formula). PCA(``n_pcs=10``) → kNN(``n_neighbors=20``).
2. **Cluster-mean CNV profile** — per-subcluster row-mean of the
   gene-level CNV matrix.
3. **Per-chromosome R-style Viterbi** — ``use_r_viterbi=True`` uses
   R's ``pnorm``-based emission with ``t = 1e-6`` and the R i6
   prior. The emission std is shrunk by cluster size:
   ``sd_ci = base_sd / sqrt(N_cells_in_cluster)``.
4. **Segment-length denoise** — collapse non-neutral segments shorter
   than ``min_segment_length`` (default 5) back to neutral. Equivalent
   to R ``denoise=TRUE``.
5. **Region-rule tumor call** — a subcluster with ≥
   ``min_segments_for_tumor`` (default 1) surviving non-neutral
   segments is labeled ``"Tumor"`` (R ``pred_cnv_regions.dat``
   semantics).
6. **Broadcast** subcluster label and per-cluster non-neutral
   window fraction to every cell.

Output (with default ``key_added="cnv_tumor_call"``):

* ``adata.obs["cnv_tumor_call"]`` — ``"Tumor"`` / ``"Normal"``
  (region-rule).
* ``adata.obs["cnv_tumor_call_score"]`` — per-cell
  *non-neutral window fraction* (algorithmic equivalent of R
  ``proportion_cnv_R``). **Use this as your per-cell HMM-based score;
  do not rebuild it externally** — see :doc:`dev_notes`.
* ``adata.obs["cnv_tumor_call_subcluster"]`` — leiden subcluster id.

Per-cell score helpers
----------------------

There are three commonly requested per-cell metrics:

+--------------------------------+----------------------------------------------------------+-----------------------------+
| Metric                         | Definition                                               | How to compute              |
+================================+==========================================================+=============================+
| ``cnv_signal``                 | ``mean(|X_cnv_r - 1|)`` per cell                         | Direct on                   |
| (= R ``cnv_signal_R``)         |                                                          | ``adata.obsm["X_cnv_r"]``   |
+--------------------------------+----------------------------------------------------------+-----------------------------+
| ``proportion_cnv``             | ``mean(state ≠ neutral)`` per cell                       | ``obs["{key_added}_score"]``|
| (= R ``proportion_cnv_R``)     |                                                          |                             |
+--------------------------------+----------------------------------------------------------+-----------------------------+
| Region-rule tumor call         | subcluster has ≥ ``min_segments_for_tumor``              | ``obs["{key_added}"]``      |
| (= R ``pred_cnv_regions.dat``) | non-neutral segments after denoise                       |                             |
+--------------------------------+----------------------------------------------------------+-----------------------------+

For a full-strength "two-metric AND-gate" tumor call (the
top-tier-journal recommended call), threshold both ``cnv_signal`` and
``proportion_cnv`` at the P95 of the **reference** distribution and
take the AND. See :doc:`dev_notes` § 1 for a copy-pasteable example.

Reference / regression suite
----------------------------

* PDAC 40-sample R-vs-cnvturbo benchmark, region-rule:
  100% concordance with R ``pred_cnv_regions.dat``.
* PDAC 3-sample strict-rule benchmark, two-metric AND-gate:
  Cohen κ ∈ {0.595, 0.816, 0.911}, MCC ∈ {0.634, 0.819, 0.914};
  per-cell ``cnv_signal`` Pearson r ∈ {0.92, 0.96, 0.98}.

These numbers are reproduced by the regression scripts under
``tests/`` and are required to remain stable across releases.
