# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-25

### Added
- `tl.infercnv_r_compat` 与 `tl.compute_hspike_emission_params` 新增 `min_mean_expr_cutoff`
  参数（默认 `0.1`），等价于 R inferCNV `require_above_min_mean_expr_cutoff`。10x 数据
  推荐 `0.1`，Smart-seq2 全长数据建议 `1.0`，传 `0` 可关闭过滤（v0.1.0 行为）。
- `adata.uns["cnv"]` 新增字段 `kept_var_names`（`obsm["X_cnv"]` 列对应的原始 `var_names`）
  与 `min_mean_expr_cutoff`（实际应用的 cutoff，便于追溯）。

### Changed
- **(behavior change)** 默认管线现在会在 Step 1（取 raw count）之后立即过滤低表达基因，
  与 R inferCNV `02_reduced_by_cutoff` 行为完全一致；`scale_factor` 与 `cell_totals`
  也基于过滤后的矩阵计算（与 R `normalize_counts_by_seq_depth` 一致）。
- 升级影响：相同输入下，`adata.obsm["X_cnv"]` 的列数从全 GTF 基因数（~26k）降到
  ~3k–8k（与 R 一致）；HMM 区域调用 100% 不变，但 cell-level `cnv_score` 的绝对
  数值会变化。如需保留 v0.1.0 行为，显式传 `min_mean_expr_cutoff=0`。

### Fixed
- 此前因 `cnvturbo` 不做 `reduce_by_cutoff`，~85% 的低表达基因被纳入归一化与滑窗平滑
  分母，使 `cnv_score`（mean(|x − 1|)）系统性偏低、与 R 在数值上不可直接对齐。
  v0.2.0 起 40 样本验证：cnv_signal Pearson r 中位 **0.85 → 0.957**（与 R 对比），
  全部 40 样本最差也 ≥ 0.89；HMM 区域调用仍 100% 一致。

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

[0.2.0]: https://github.com/LogicByteCraft/cnvturbo/releases/tag/v0.2.0
[0.1.0]: https://github.com/LogicByteCraft/cnvturbo/releases/tag/v0.1.0

