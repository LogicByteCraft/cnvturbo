# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `tl.infercnv_r_compat` 与 `tl.compute_hspike_emission_params` 新增 `min_mean_expr_cutoff`
  参数（默认 `0.1`），等价于 R inferCNV `require_above_min_mean_expr_cutoff`。10x 数据
  推荐 `0.1`，Smart-seq2 全长数据建议 `1.0`，传 `0` 可关闭过滤（旧版行为）。
- `adata.uns["cnv"]` 新增字段 `kept_var_names`（`obsm["X_cnv"]` 列对应的原始 `var_names`）
  与 `min_mean_expr_cutoff`（实际应用的 cutoff，便于追溯）。

### Changed
- 默认管线现在会在 Step 1（取 raw count）之后立即过滤低表达基因，与 R inferCNV
  `02_reduced_by_cutoff` 的行为完全一致；`scale_factor` 与 `cell_totals` 也基于
  过滤后的矩阵计算（与 R `normalize_counts_by_seq_depth` 一致）。这显著降低了
  `cnv_score` 在绝对数值上与 R 的偏差（HMM 区域调用本来已 100% 一致）。

### Fixed
- 此前因 `cnvturbo` 不做 `reduce_by_cutoff`，导致 ~85% 的低表达基因被纳入归一化与
  滑窗平滑分母，使 `cnv_score`（cell-level absolute deviation）系统性偏低、与 R 在数
  值上不可直接对齐。0.1 起此问题修复，新参数默认开启。

## [0.1.0] - 2026-04-24

### Added
- Initial public release of `cnvturbo`.
- `tl.infercnv_r_compat`: gene-space, copy-ratio R inferCNV-equivalent 8-step pipeline.
- `tl.compute_hspike_emission_params`: hspike-based HMM emission calibration.
- `tl.hmm_call_subclusters`: cell-level Tumor / Normal classification via per-subcluster HMM i6 (R-equivalent Viterbi + denoise).
- `tl.hmm_call_cells`: cell-level HMM caller without subclustering.
- Numba (CPU) and PyTorch (GPU) accelerated sliding-window convolution and batched Viterbi.
- Continuous cell-level CNV burden score `cnv_call_score`.
- Verified 100% cell-level concordance with R inferCNV's HMM output on 3 PDAC samples (15,135 cells), at 100–230× speed-up.

[0.1.0]: https://github.com/LogicByteCraft/cnvturbo/releases/tag/v0.1.0

