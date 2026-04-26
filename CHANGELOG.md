# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 新增 `template/` 调用示例目录，提供 R-compatible 分析、可视化，以及可选的
  R inferCNV 对照 benchmark 脚本，方便外部用户按 AnnData + GTF 独立运行
  `cnvturbo`，不依赖 R inferCNV 结果。
- `io.genomic_position_from_gtf(gtf_gene_id="auto")` 支持先按 `gene_name`
  匹配，再用去版本号的 `gene_id` 补齐未匹配基因，并保留 GTF 染色体顺序。
- 新增 `tl.denoise_r_compat`，移植 R inferCNV Step 22
  `clear_noise_via_ref_mean_sd`，用于计算与 R `cnv_signal_R` 对齐的
  strict 连续信号。

### Changed
- `tl.infercnv_r_compat` 将 no-coordinate / excluded chromosome / low-expression
  基因过滤前移到 library-size normalization 之前，对齐 R inferCNV 初始对象的
  gene pool 语义；染色体内排序改为稳定排序，避免同起点基因顺序漂移。
- `tl.compute_hspike_emission_params` 对齐 R `sim_method="meanvar"` hidden-spike
  模拟，返回可选 SD trend 参数供 subcluster HMM 按细胞数缩放 emission 方差。
- `tl.hmm_call_subclusters` 对齐 R inferCNV subcluster 逻辑：按 annotation group
  分别 Leiden、支持 R-style z-score gene filter、CPM/Seurat SNN 图构建，并将
  `{key_added}_score` 定义为 HMM non-neutral state fraction。
- README benchmark 更新为 40 样本验证结果：region-level CNV 100% 一致，
  strict cell-level F1=0.980，per-cell `cnv_score` mean Pearson=0.99997。
- 移除继承自 infercnvpy 风格的 legacy pytest 测试目录与 pytest 配置，避免将旧
  `tl.infercnv` 测试误解为当前 R-compatible 主路径的验证依据。

### Documentation
- 修订 `README.md` 与 `pyproject.toml` 中关于 GPU / Numba 加速的描述，明确后端
  覆盖范围，避免读者误以为 R-兼容主路径享受 GPU 加速：
    * 头部 hero 段落、对比表 "GPU / Numba acceleration" 行、Cell-level
      concordance 行（infercnvpy 由占位的 `0.81` 改为 `N/A (no cell-level HMM)`）、
      Benchmark 节、Optional Requirements、Design highlights、BibTeX title 全部
      标注或拆分了 "**legacy `tl.infercnv` + `tl.hmm_call_cells` 路径走 GPU/Numba**"
      与 "**R-exact 主路径目前仅 CPU + joblib**"。
    * 新增 §Backend coverage 子节，逐函数列出 Numba CPU / PyTorch CUDA 支持
      矩阵，并备注 `tl.hmm_call_subclusters` 的 `backend` 形参在
      `use_r_viterbi=True`（默认）时是 no-op。
    * 在 `[project.optional-dependencies]` 注释里注明这些 extras 不会加速 R-exact
      主路径；`pyproject.toml` description 同步更新。
- 重写 `docs/infercnv.rst`：去掉早期 `cnvturbo.tl.infercnv`（log-fold-change /
  lfc_cap / dynamic_threshold 那一套）的描述，改为只覆盖 R-兼容主线流程
  （`infercnv_r_compat` + `compute_hspike_emission_params` + `hmm_call_subclusters`
  三段式），并补上输入数据格式、各阶段输出键、per-cell score 取用表，与 R
  对齐参数 / 顶刊 strict 双门控示例。
- 精简 `docs/api.rst` 到 R-兼容工作流的核心 API 表面：去掉 `copykat` /
  `ithcna` / `ithgex` / `pca` / `umap` / `tsne` / `chromosome_heatmap_summary` /
  `leiden` / `read_scevan` / `pp.neighbors` 等旧 infercnvpy 风格条目（函数本身
  仍存在，只是不再列入生成的参考页）；新增 `infercnv_r_compat` /
  `compute_hspike_emission_params` / `hmm_call_subclusters` / `hmm_call_cells` /
  `cnv_score_cell` 章节。
- 新增 `docs/dev_notes.md`，沉淀四节非显然教训：
    1. **不要在外部脚本里重建 cluster-level Viterbi**——记录 `sd_ci = base_sd /
       sqrt(n_ci)` 在大 reference cluster 上退化的失败模式与正确做法（直接用
       `obs["{key_added}_score"]`），附 PDAC 3 样本 strict κ ∈ {0.595, 0.816,
       0.911} 的 regression baseline。
    2. `cnv_score_cell` 的 HMM-based / fallback 双路径陷阱（`neutral_state=2`
       默认与 R i6 不符；fallback 路径 `mean(|X|)` 漏减 1.0）。
    3. v0.2.1 GTF Categorical 修复历史 + workaround。
    4. 与 R inferCNV 默认值对齐的 9 项关键参数对照表（含 Smart-seq2 注意事项）。
- 更新 `docs/index.md` toctree 加入 `dev_notes.md`。

## [0.2.1] - 2026-04-25

### Fixed
- `io.genomic_position_from_gtf` 在 `gtfparse >= 2.0` / pyarrow backend 下抛
  `TypeError: can only concatenate str (not "Categorical") to str` 的问题。
  根因：`gtfparse` 新版本默认返回 Categorical / ArrowExtensionArray 列，下游
  `"chr" + col`、`col.str.startswith(...)`、`col.str.replace(...)` 都假设 object/str
  dtype。修复方式是在拿到 GTF 后立即把 `chromosome` / `gene_id` / `gene_name` 三列
  显式 `.astype(str)`，与旧版 gtfparse 的行为完全等价，不影响已正常工作的环境。
  影响：v0.2.0 用户在新装的 `gtfparse` 环境下基本必然踩到此 bug，无需任何 workaround
  （如手动重写 GTF 解析）即可升级到 v0.2.1 直接调用。
- `io.genomic_position_from_biomart` 在 chromosome 字段被 cache/反序列化为
  Categorical 时也存在同款隐患，一并加固。
- 新增回归测试 `tests/test_io.py::test_get_genomic_position_from_gtf_categorical_chromosome`，
  通过 mock `gtfparse.read_gtf` 强制返回 Categorical 列覆盖该退化场景。

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

