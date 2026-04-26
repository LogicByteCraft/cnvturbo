"""R inferCNV 兼容管线：严格按照 R inferCNV 源码（inferCNV_ops.R）复现分析流程。

R inferCNV 真实管线（基于源码分析 infercnv-master/R/inferCNV_ops.R）
--------------------------------------------------------------------
全程在 **gene × cell** 维度运行（与 R `infercnv_obj@expr.data` 一致）：

Step 1  : 库容归一化（÷ colSums × median(colSums)）
Step 2  : log2(x + 1)
Step 3  : 第一次参考减法（bounds 模式：落在参考 [min, max] 区间内置 0；超出则减边界）
Step 4  : clip 到 ±max_centered_threshold（默认 3）
Step 5  : 按染色体 same-length 滑窗平滑（window=101；pyramidinal 加权卷积 + 端点修正）
          —— 输出 (n_cells, n_genes_filt)，**与输入基因数完全一致**
Step 6  : per-cell MEDIAN 中心化（center_cell_expr_across_chromosome, method="median"）
Step 7  : 第二次参考减法（bounds 模式，仍在 gene 维：subtract_ref_expr_from_obs 第二次调用）
Step 8  : 2^x 变换（invert_log2，将 log2 残差转为 copy-ratio 空间，中性态 ≈ 1.0）

HMM 在 copy-ratio 空间（≥ 0）的 **基因级序列**上运行；中性态期望值 = 1.0。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm

from cnvturbo.tl._backend import get_n_jobs
from cnvturbo.tl._infercnv import (
    _ordered_chromosomes,
    _running_mean_same_length_by_chromosome,
)


# ── 公开 API ──────────────────────────────────────────────────────────────────


def infercnv_r_compat(
    adata: AnnData,
    *,
    raw_layer: str = "counts",
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    max_ref_threshold: float = 3.0,
    window_size: int = 101,
    exclude_chromosomes: Sequence[str] | None = ("chrM", "chrMT", "MT", "M"),
    min_mean_expr_cutoff: float = 0.1,
    n_jobs: int | None = None,
    inplace: bool = True,
    key_added: str = "cnv",
    apply_2x_transform: bool = True,
) -> None | tuple[dict, scipy.sparse.csr_matrix]:
    """从原始 count 矩阵严格复现 R inferCNV 分析管线（全程基因维）。

    与 R inferCNV (`infercnv::run`) 的步骤一一对应（HMM 模式 i6, subclusters）：
      Step 1: 库容归一化（→ median 库深）
      Step 2: log2(x + 1)
      Step 3: 第一次 bounds 模式参考减法（gene 维）
      Step 4: clip 到 ±max_ref_threshold
      Step 5: 按染色体 same-length 滑窗平滑（输出 = 输入基因数 + 端点修正）
      Step 6: per-cell MEDIAN 中心化（滑窗之后）
      Step 7: 第二次 bounds 模式参考减法（**仍在 gene 维**）
      Step 8: 2^x 变换 → copy-ratio 空间（中性态 ≈ 1.0）

    重要变更（v2 R-exact 对齐）：
      - 全程在 (n_cells, n_genes_filt) 维度，不做窗口降采样
      - chr_pos 存储 **基因级染色体起始下标**（非窗口下标）
      - 默认 apply_2x_transform=True，HMM 输入空间为 copy-ratio

    Parameters
    ----------
    adata
        包含原始 count 的 AnnData。var 须有 chromosome / start / end 列。
    raw_layer
        存放原始整数 count 的 layer 名。若不存在则使用 adata.X。
    reference_key
        obs 中标记细胞类型的列名（如 "infercnv_group_type"）。
    reference_cat
        参考细胞的类别值（如 "Reference" 或 ["NK", "Endo"]）。
    max_ref_threshold
        参考减法后的 clip 阈值（R 默认 3）。
    window_size
        滑窗大小（R 默认 101；奇数）。
    exclude_chromosomes
        排除的染色体（默认仅排除线粒体染色体，保留 chrX/chrY）。
    min_mean_expr_cutoff
        与 R inferCNV 的 ``require_above_min_mean_expr_cutoff`` 等价：
        过滤所有 ``mean(raw_count) < cutoff`` 的低表达基因。
        10x 数据 R 默认 0.1；Smart-seq2 全长数据建议 1.0；
        设为 0 可完全关闭过滤（与旧版行为一致，但不推荐）。
        过滤后基因数通常会减少到原 var 的 10%~20%，与 R 行为一致。
    n_jobs
        CPU 线程数；None 使用全部核心。
    inplace
        True 将结果写入 adata；False 返回 (chr_pos, cnv_matrix)。
    key_added
        结果键名（adata.obsm["X_{key_added}"] 和 adata.uns[key_added]）。
    apply_2x_transform
        True 在最后做 2^x 变换（copy-ratio 空间，与 R HMM 输入一致）；
        False 保留 log2 残差（兼容旧版接口）。
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns):
        raise ValueError(
            "Genomic positions not found. "
            "There need to be `chromosome`, `start`, and `end` columns in `adata.var`."
        )

    n_jobs_eff = get_n_jobs(n_jobs)

    # ── Step 1：获取原始 count ────────────────────────────────────────────────
    if raw_layer in adata.layers:
        raw_counts = adata.layers[raw_layer]
        logging.info(f"infercnv_r_compat: using layer '{raw_layer}'")  # type: ignore
    else:
        raw_counts = adata.X
        logging.warning(  # type: ignore
            f"Layer '{raw_layer}' not found; falling back to adata.X. "
            "Ensure raw (unnormalized) counts are provided."
        )
    if scipy.sparse.issparse(raw_counts):
        raw_counts = raw_counts.toarray()
    raw_counts = np.asarray(raw_counts, dtype=np.float64)

    # ── Step 1.25：先限定到 R gene_order 基因池 ───────────────────────────
    # R inferCNV 在 run() 之前的 infercnv_obj 已经只包含 gene_ordering_file 中
    # 的基因；低表达过滤和 library-size 归一化都发生在这个基因池内。这里必须
    # 先裁掉无坐标/排除染色体，否则全转录组库容会把 Step14 信号系统性压低。
    var_full = adata.var[["chromosome", "start", "end"]]
    var_mask_null = var_full["chromosome"].isnull().values
    var_mask_excl = (
        var_full["chromosome"].isin(exclude_chromosomes).values
        if exclude_chromosomes is not None
        else np.zeros(var_full.shape[0], dtype=bool)
    )
    gene_order_mask = ~(var_mask_null | var_mask_excl)
    raw_counts = raw_counts[:, gene_order_mask]
    var_gene_order = var_full.loc[gene_order_mask, :]
    logging.info(  # type: ignore
        f"  gene_order-compatible genes: {raw_counts.shape[1]} / {adata.n_vars}"
    )

    # ── Step 1.5：低表达基因过滤（R `require_above_min_mean_expr_cutoff`）──
    # R 中此步在归一化之前执行；后续 cell_totals 用的是过滤后的矩阵和。
    # 不能在归一化后再做，否则 cell_totals 会包含被过滤掉的低表达基因贡献。
    if min_mean_expr_cutoff is not None and min_mean_expr_cutoff > 0:
        gene_raw_means = raw_counts.mean(axis=0)
        lowexpr_mask = gene_raw_means < float(min_mean_expr_cutoff)
        n_filtered = int(lowexpr_mask.sum())
        if n_filtered > 0:
            raw_counts = raw_counts[:, ~lowexpr_mask]
            var_gene_order = var_gene_order.loc[~lowexpr_mask, :]
        logging.info(  # type: ignore
            f"  reduce_by_cutoff: filtered {n_filtered} / {gene_order_mask.sum()} genes "
            f"(mean raw count < {min_mean_expr_cutoff}); kept {raw_counts.shape[1]} genes"
        )

    # ── Step 2：库容归一化 → log2(x+1) ───────────────────────────────────────
    # cell_totals 基于过滤后的 raw_counts，与 R `normalize_counts_by_seq_depth` 一致
    cell_totals = np.maximum(raw_counts.sum(axis=1, keepdims=True), 1.0)
    scale_factor = float(np.median(raw_counts.sum(axis=1)))
    scale_factor = max(scale_factor, 1.0)
    normalized = raw_counts / cell_totals * scale_factor
    log_expr = np.log1p(normalized) / math.log(2)  # log2(x + 1)
    logging.info(  # type: ignore
        f"  Step 2 done: log_expr mean={log_expr.mean():.4f}, scale_factor={scale_factor:.0f}"
    )

    # ── 解析参考细胞掩码 ──────────────────────────────────────────────────────
    ref_mask = _get_ref_mask(adata, reference_key, reference_cat)
    logging.info(f"  Reference cells: {ref_mask.sum()} / {adata.n_obs}")  # type: ignore

    # ── Step 3：第一次 bounds 模式参考减法（gene 维）─────────────────────────
    log_expr = _subtract_ref_bounds(log_expr, ref_mask)
    logging.info(  # type: ignore
        f"  Step 3 (1st ref subtract, gene-space) done: "
        f"mean={log_expr.mean():.5f}, std={log_expr.std():.5f}"
    )

    # ── Step 4：clip ──────────────────────────────────────────────────────────
    log_expr = np.clip(log_expr, -max_ref_threshold, max_ref_threshold).astype(np.float32)

    # ── 按 R gene_order 顺序重排基因 ────────────────────────────────────────
    expr_filt = log_expr
    var_filt = var_gene_order
    # 后续平滑会按 chromosome/start 重排基因；这里先同步重排矩阵和 var，
    # 确保 X_{key_added} 的列顺序与 kept_var_names 完全一致。
    ordered_genes: list[str] = []
    for chrom in _ordered_chromosomes(var_filt):
        chrom_genes = (
            var_filt.loc[var_filt["chromosome"] == chrom]
            .sort_values("start", kind="mergesort")
            .index.to_list()
        )
        ordered_genes.extend(chrom_genes)
    order_idx = var_filt.index.get_indexer(ordered_genes)
    expr_filt = expr_filt[:, order_idx]
    var_filt = var_filt.loc[ordered_genes, :]
    logging.info(f"  Genes for smoothing: {expr_filt.shape[1]} / {adata.n_vars}")  # type: ignore

    # ── Step 5：按染色体 same-length 滑窗平滑（gene 维输出） ─────────────────
    # 与 R smooth_by_chromosome 一致：每条染色体独立平滑，输出长度 = 该染色体基因数
    logging.info(  # type: ignore
        f"  Step 5: same-length sliding window (size={window_size}, n_jobs={n_jobs_eff})"
    )

    chunk_size = max(200, adata.n_obs // max(n_jobs_eff, 1))
    cell_chunks = list(enumerate(
        expr_filt[i : i + chunk_size, :]
        for i in range(0, adata.n_obs, chunk_size)
    ))

    results: dict[int, tuple[dict, np.ndarray]] = {}

    def _smooth_chunk(chunk_idx: int, chunk: np.ndarray):
        cp, smoothed = _running_mean_same_length_by_chromosome(
            chunk, var_filt, window_size=window_size,
        )
        return chunk_idx, cp, smoothed

    with ThreadPoolExecutor(max_workers=n_jobs_eff) as pool:
        futures = {pool.submit(_smooth_chunk, ci, ch): ci for ci, ch in cell_chunks}
        for fut in tqdm(
            as_completed(futures), total=len(cell_chunks), desc="infercnv_r_compat"
        ):
            chunk_idx, cp, smoothed = fut.result()
            results[chunk_idx] = (cp, smoothed)

    chr_pos = results[0][0]  # 基因级染色体起始下标
    x_smoothed = np.vstack([results[i][1] for i in range(len(cell_chunks))])  # (n_cells, n_genes_filt)
    logging.info(  # type: ignore
        f"  Step 5 done: smoothed shape={x_smoothed.shape} (gene-level)"
    )

    # ── Step 6：per-cell MEDIAN 中心化（R: center_cell_expr_across_chromosome）
    cell_medians = np.median(x_smoothed, axis=1, keepdims=True)
    x_centered = x_smoothed - cell_medians
    logging.info(  # type: ignore
        f"  Step 6 (per-cell median center) done: "
        f"residual mean={x_centered.mean():.6f}, std={x_centered.std():.5f}"
    )

    # ── Step 7：第二次 bounds 模式参考减法（gene 维，与 R 一致）──────────────
    # R `subtract_ref_expr_from_obs` 第二次调用，仍在基因维度
    x_centered = _subtract_ref_bounds_matrix(x_centered, ref_mask)
    logging.info(  # type: ignore
        f"  Step 7 (2nd ref subtract, gene-space) done: "
        f"mean={x_centered.mean():.6f}, std={x_centered.std():.5f}"
    )

    # ── Step 8：2^x 变换 → copy-ratio 空间（R Step: invert_log2） ────────────
    if apply_2x_transform:
        x_final = np.power(2.0, x_centered, dtype=np.float32)
        logging.info(  # type: ignore
            f"  Step 8 (2^x): copy-ratio mean={x_final.mean():.4f}, "
            f"ref mean={x_final[ref_mask].mean():.4f}, neutral expected=1.0"
        )
    else:
        x_final = x_centered.astype(np.float32)

    ref_anchor = x_final[ref_mask].mean()
    logging.info(  # type: ignore
        f"  Ref anchor mean={ref_anchor:.4f} "
        f"(expected ≈1.0 if apply_2x_transform=True, ≈0.0 if False)"
    )

    res = scipy.sparse.csr_matrix(x_final)

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        # chr_pos: 基因级染色体起始下标；is_gene_space: 标记输入空间维度
        # kept_var_names: X_{key_added} 的列对应的原始 var_names（已经过 reduce_by_cutoff
        # + chromosome null/exclude 过滤），方便下游 join 回 adata.var。
        adata.uns[key_added] = {
            "chr_pos": chr_pos,
            "is_gene_space": True,
            "is_copy_ratio": bool(apply_2x_transform),
            "kept_var_names": var_filt.index.to_list(),
            "min_mean_expr_cutoff": float(min_mean_expr_cutoff or 0.0),
        }
        return None
    else:
        return chr_pos, res


def denoise_r_compat(
    cnv_matrix: np.ndarray | scipy.sparse.spmatrix,
    ref_mask: np.ndarray,
    *,
    sd_amplifier: float = 1.5,
    noise_logistic: bool = False,
) -> np.ndarray:
    """移植 R inferCNV Step 22 `clear_noise_via_ref_mean_sd`。

    R strict cnv_signal 来自 22_denoise infercnv_obj：
    `colMeans(abs(expr.data - 1))`。HMM proportion 来自 17_HMM_pred，
    因此这个 denoise 矩阵只用于 continuous score，不回写给 HMM 输入。
    """
    if noise_logistic:
        raise NotImplementedError(
            "R-compatible logistic denoise is not implemented; "
            "inferCNV default noise_logistic=FALSE uses hard replacement."
        )

    x = cnv_matrix.toarray() if scipy.sparse.issparse(cnv_matrix) else np.asarray(cnv_matrix)
    x = np.asarray(x, dtype=np.float64).copy()
    ref_mask = np.asarray(ref_mask, dtype=bool)
    if x.ndim != 2:
        raise ValueError("cnv_matrix must be a 2D matrix")
    if ref_mask.shape[0] != x.shape[0]:
        raise ValueError("ref_mask length must match cnv_matrix rows (cells)")
    if not ref_mask.any():
        raise ValueError("ref_mask contains no reference cells")

    ref_vals = x[ref_mask, :]
    mean_ref_vals = float(np.mean(ref_vals))
    mean_ref_sd = float(np.mean(np.std(ref_vals, axis=1, ddof=1)) * sd_amplifier)
    lower_bound = mean_ref_vals - mean_ref_sd
    upper_bound = mean_ref_vals + mean_ref_sd

    in_noise = (x > lower_bound) & (x < upper_bound)
    x[in_noise] = mean_ref_vals
    return x


# ── 内部函数 ──────────────────────────────────────────────────────────────────


def _get_ref_mask(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: str | Sequence[str] | None,
) -> np.ndarray:
    """返回参考细胞布尔掩码（True = 参考细胞）。"""
    if reference_key is None or reference_cat is None or reference_key not in adata.obs.columns:
        logging.warning(  # type: ignore
            "reference_key / reference_cat not specified; using ALL cells as reference. "
            "Provide normal cell annotations for meaningful CNV detection."
        )
        return np.ones(adata.n_obs, dtype=bool)
    cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
    mask = adata.obs[reference_key].isin(cats).values
    if mask.sum() < 10:
        logging.warning(f"Only {mask.sum()} reference cells found!")  # type: ignore
    return mask


def _subtract_ref_bounds(
    log_expr: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """R Step 4 / Step 8（基因空间）：bounds 模式参考减法。

    R 源码 inferCNV_ops.R ~1759–1774 行（ref_subtract_use_mean_bounds=TRUE）：
      - 单参考组：每个基因，x < ref_mean 时 x - ref_mean；x >= ref_mean 时不变（常数截断）。
      实际等价于：多组取 [grp_min, grp_max]；单组 min=max=mean。
      - 落在 [ref_min, ref_max] 内 → 置 0；超出 → 减对应边界。

    单参考组简化：
      delta = log_expr - ref_mean_per_gene
      落在 [-∞, 0]：置 0（肿瘤细胞低于参考均值的基因 → 不认为是真实损失，仅保留超出部分）
      实际 R 逻辑（use_bounds=TRUE, 单组）：
        grp_min = grp_max = mean(ref)
        → 所有 obs 值减去 ref_mean；超出 0 → x - ref_mean；低于 0 → 0
      等价于：max(x - ref_mean, 0) 对 x > ref_mean；min(x - ref_mean, 0) 对 x < ref_mean
      即：x_centered = x - ref_mean，然后保持原始符号
    即与简单 mean 减法完全等价（单参考组时 use_bounds 退化为简单减法）。
    """
    ref_mean = log_expr[ref_mask, :].mean(axis=0, keepdims=True)   # (1, n_genes)
    return log_expr - ref_mean


def _subtract_ref_bounds_matrix(
    x: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """R Step 12（窗口空间）：第二次参考减法（per-window）。

    作用在滑窗平滑 + per-cell median 中心化之后的矩阵上。
    参考细胞此时已经过 median 中心化，对每个窗口再减参考细胞均值，
    使参考细胞最终锚定在 0（log2 空间）/ 1.0（copy-ratio 空间）。
    """
    ref_window_mean = x[ref_mask, :].mean(axis=0, keepdims=True)   # (1, n_windows)
    return x - ref_window_mean


# ── hspike 精确实现（复现 R inferCNV_hidden_spike.R）──────────────────────────


def compute_hspike_emission_params(
    adata: AnnData,
    *,
    raw_layer: str = "counts",
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    max_ref_threshold: float = 3.0,
    window_size: int = 101,
    exclude_chromosomes: Sequence[str] | None = ("chrM", "chrMT", "MT", "M"),
    min_mean_expr_cutoff: float = 0.1,
    n_sim_cells: int = 100,
    n_genes_per_chr: int = 400,
    cnv_ratios: tuple[float, ...] = (0.01, 0.5, 1.0, 1.5, 2.0, 3.0),
    output_space: str = "copy_ratio",
    random_state: int = 42,
    return_sd_trend: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """精确复现 R inferCNV hspike 模拟，计算 HMM 发射参数（单细胞水平）。

    v2 关键变更：
      - 全程使用与真实数据同样的 8 步管线（含 same-length 平滑、第二次参考减法在 gene 维）
      - 新增 `output_space="copy_ratio"`：emission 参数输出在 copy-ratio 空间（中性 ≈ 1.0），
        与新的 HMM 输入空间（infercnv_r_compat 的 apply_2x_transform=True）一致

    与 R inferCNV_hidden_spike.R 的对应关系
    ----------------------------------------
    R 逻辑（精炼版）：
      1. 构建合成基因组（交替中性/CNV 染色体，每条 n_genes_per_chr 基因）
      2. 从真实参考细胞中带放回采样 → 合成正常 + 肿瘤细胞
      3. 对合成细胞跑完整 inferCNV 管线（与真实数据相同的 8 步）
      4. 从各 CNV 染色体提取均值/方差 → emission 参数

    Returns
    -------
    emit_means : (n_states,) float64 — 各 CNV 状态的 emission 均值（copy_ratio 或 log2 空间）
    emit_stds  : (n_states,) float64 — 各 CNV 状态的 emission 标准差（单细胞水平）
    sd_intercepts / sd_slopes
        当 return_sd_trend=True 时额外返回，对应 R
        lm(log(sd_vals) ~ log(num_cells)) 的截距与斜率。
    """
    import pandas as pd  # noqa: PLC0415

    rng = np.random.default_rng(random_state)

    # ── 获取参考细胞基因均值（用于构建合成基因组）──────────────────────────
    ref_mask = _get_ref_mask(adata, reference_key, reference_cat)
    if raw_layer in adata.layers:
        raw_counts = adata.layers[raw_layer]
    else:
        raw_counts = adata.X
    if scipy.sparse.issparse(raw_counts):
        raw_counts = raw_counts.toarray()
    raw_counts = np.asarray(raw_counts, dtype=np.float64)

    var_mask_excl = (
        adata.var["chromosome"].isin(list(exclude_chromosomes))
        if exclude_chromosomes is not None
        else pd.Series(False, index=adata.var_names)
    )
    gene_order_mask = ~(adata.var["chromosome"].isnull() | var_mask_excl)
    raw_gene_order = raw_counts[:, gene_order_mask.values]

    # 与真实管线保持一致：先限定到 R gene_order 基因池，再做低表达过滤和库容归一化。
    if min_mean_expr_cutoff is not None and min_mean_expr_cutoff > 0:
        gene_raw_means = raw_gene_order.mean(axis=0)
        keep_after_cutoff = gene_raw_means >= float(min_mean_expr_cutoff)
    else:
        keep_after_cutoff = np.ones(raw_gene_order.shape[1], dtype=bool)

    raw_pool = raw_gene_order[:, keep_after_cutoff]
    pool_totals = np.maximum(raw_pool.sum(axis=1, keepdims=True), 1.0)
    pool_scale_factor = max(float(np.median(raw_pool.sum(axis=1))), 1.0)
    norm_pool = raw_pool / pool_totals * pool_scale_factor

    # hspike 的 gene means 来自已归一化 infercnv_obj@expr.data；这里再限制到
    # 后续真实 HMM 会使用的有效染色体基因池，避免用无坐标/排除染色体基因估计发射分布。
    norm_keep = norm_pool
    n_genes_total = int(norm_keep.shape[1])

    # ── 构建合成基因组（与 R .get_hspike_chr_info 一致）────────────────────
    cnv_chr_defs = [
        ("chr_0",    0.01),
        ("chr_0pt5", 0.50),
        ("chr_1pt5", 1.50),
        ("chr_2",    2.00),
        ("chr_3pt0", 3.00),
    ]
    neutral_chr_names = ["chrA", "chrB", "chrC", "chrD", "chrE"]

    n_non_f = (len(cnv_chr_defs) + len(neutral_chr_names)) * n_genes_per_chr
    n_chr_f = max(n_genes_total - n_non_f, n_genes_per_chr)
    n_synthetic_genes = n_non_f + n_chr_f
    sampled_idx = rng.integers(0, n_genes_total, size=n_synthetic_genes)

    chr_struct: list[tuple[str, float, int]] = []
    for ni, nname in enumerate(neutral_chr_names):
        chr_struct.append((nname, 1.0, n_genes_per_chr))
        if ni < len(cnv_chr_defs):
            chr_struct.append(cnv_chr_defs[ni] + (n_genes_per_chr,))
    chr_struct.append(("chr_F", 1.0, n_chr_f))

    sim_var_rows = []
    for chr_name, _, n_g in chr_struct:
        for i in range(n_g):
            sim_var_rows.append({"chromosome": chr_name, "start": i + 1, "end": i + 2})
    sim_var = pd.DataFrame(sim_var_rows)
    sim_var.index = [f"hspike_gene_{i}" for i in range(len(sim_var))]

    chr_gene_offset = 0
    gene_ratio_vec = np.ones(n_synthetic_genes, dtype=np.float64)
    for chr_name, ratio, n_g in chr_struct:
        if ratio != 1.0:
            gene_ratio_vec[chr_gene_offset: chr_gene_offset + n_g] = ratio
        chr_gene_offset += n_g

    cnv_chr_names = {c for c, r, _ in chr_struct if r != 1.0}

    logging.info(  # type: ignore
        f"compute_hspike_emission_params: n_synthetic_genes={n_synthetic_genes}, "
        f"n_sim_cells={n_sim_cells}, cnv_chrs={sorted(cnv_chr_names)}, "
        f"output_space={output_space}"
    )

    # ── R sim_method='meanvar'：用 mean-variance trend 重新模拟 counts ─────────
    # R inferCNV_hidden_spike.R:
    #   gene_means <- rowMeans(normal_cells_expr)[sampled_genes]
    #   sim_normal <- .get_simulated_cell_matrix_using_meanvar_trend(..., include.dropout=TRUE)
    #   sim_tumor  <- same mean-var simulator with hspike_gene_means = gene_means * cnv
    #
    # 之前的 Python 版本直接重采样真实 reference counts，会低估 hspike 的模拟方差，
    # 使 HMM emission 与 R 的 i6 hspike 不同。这里按 R 默认 meanvar 路径移植。
    from scipy.interpolate import UnivariateSpline  # noqa: PLC0415

    ref_norm_keep = norm_keep[ref_mask]
    ref_gene_means = ref_norm_keep.mean(axis=0)
    gene_means = ref_gene_means[sampled_idx].astype(np.float64)
    gene_means[gene_means == 0] = 1e-3

    if reference_key is not None and reference_key in adata.obs.columns:
        group_values = adata.obs[reference_key].astype(str)
        group_names = list(dict.fromkeys(group_values.to_list()))
        group_masks = [(group_values == g).values for g in group_names]
    else:
        group_masks = [np.ones(adata.n_obs, dtype=bool)]

    mean_vals: list[np.ndarray] = []
    var_vals: list[np.ndarray] = []
    p0_means: list[np.ndarray] = []
    p0_vals: list[np.ndarray] = []
    for gm in group_masks:
        grp = norm_keep[gm]
        if grp.shape[0] == 0:
            continue
        mean_vals.append(grp.mean(axis=0))
        var_vals.append(grp.var(axis=0, ddof=1) if grp.shape[0] > 1 else np.zeros(grp.shape[1]))
        p0_means.append(grp.mean(axis=0))
        p0_vals.append((grp == 0).mean(axis=0))

    mean_all = np.concatenate(mean_vals)
    var_all = np.concatenate(var_vals)
    x_mv = np.log(mean_all + 1.0)
    y_mv = np.log(np.maximum(var_all, 0.0) + 1.0)
    finite_mv = np.isfinite(x_mv) & np.isfinite(y_mv)
    order_mv = np.argsort(x_mv[finite_mv])
    x_mv = x_mv[finite_mv][order_mv]
    y_mv = y_mv[finite_mv][order_mv]
    # smooth.spline 的精确 GCV 在 scipy 中没有一一对应实现；UnivariateSpline 的
    # 平滑样条保留同一算法意图（log(var+1) ~ log(mean+1)）。
    k_mv = min(3, max(1, len(np.unique(x_mv)) - 1))
    mean_var_spline = UnivariateSpline(x_mv, y_mv, k=k_mv, s=len(x_mv))

    p0_mean_all = np.concatenate(p0_means)
    y_p0_all = np.concatenate(p0_vals)
    finite_p0 = np.isfinite(p0_mean_all) & np.isfinite(y_p0_all) & (p0_mean_all > 0)
    x_p0 = np.log(p0_mean_all[finite_p0])
    y_p0 = y_p0_all[finite_p0]
    order_p0 = np.argsort(x_p0)
    x_p0 = x_p0[order_p0]
    y_p0 = y_p0[order_p0]
    if len(np.unique(x_p0)) >= 2:
        k_p0 = min(3, max(1, len(np.unique(x_p0)) - 1))
        dropout_spline = UnivariateSpline(x_p0, y_p0, k=k_p0, s=len(x_p0))
    else:
        dropout_spline = None

    def _simulate_meanvar_counts(means: np.ndarray) -> np.ndarray:
        means = np.asarray(means, dtype=np.float64)
        logm = np.log(means + 1.0)
        pred_log_var = mean_var_spline(logm)
        var = np.maximum(np.exp(pred_log_var) - 1.0, 0.0)
        sim = rng.normal(
            loc=means[None, :],
            scale=np.sqrt(var)[None, :],
            size=(n_sim_cells, means.size),
        )
        sim = np.rint(np.maximum(sim, 0.0)).astype(np.float64)

        if dropout_spline is not None:
            gene_mean = sim.mean(axis=0)
            dropout_prob = np.zeros(means.size, dtype=np.float64)
            positive_mean = gene_mean > 0
            if positive_mean.any():
                dropout_prob[positive_mean] = np.asarray(
                    dropout_spline(np.log(gene_mean[positive_mean])),
                    dtype=np.float64,
                )
            dropout_prob = np.clip(np.nan_to_num(dropout_prob, nan=0.0), 0.0, 1.0)
            n_zero = (sim == 0).sum(axis=0)
            n_remaining = n_sim_cells - n_zero
            padj = np.zeros(means.size, dtype=np.float64)
            valid = n_remaining > 0
            padj[valid] = ((dropout_prob[valid] * n_sim_cells) - n_zero[valid]) / n_remaining[valid]
            padj = np.clip(np.nan_to_num(padj, nan=0.0), 0.0, 1.0)
            drop_mask = rng.random(sim.shape) <= padj[None, :]
            sim[drop_mask] = 0.0
        return sim

    sim_norm = _simulate_meanvar_counts(gene_means)
    sim_tumor = _simulate_meanvar_counts(gene_means * gene_ratio_vec)

    all_sim = np.vstack([sim_norm, sim_tumor])
    is_norm = np.array([True] * n_sim_cells + [False] * n_sim_cells)

    # ── 完整 R 管线（8 步，与 infercnv_r_compat 一致）──────────────────────
    totals  = np.maximum(all_sim.sum(axis=1, keepdims=True), 1.0)
    # R: normalize_counts_by_seq_depth(.hspike, median(colSums(normal_cells_expr)))
    # 其中 normal_cells_expr 已是原 infercnv_obj 归一化后的参考表达矩阵。
    sf      = pool_scale_factor
    sim_log = np.log1p(all_sim / totals * sf) / math.log(2)

    sim_log = _subtract_ref_bounds(sim_log, is_norm)
    sim_log = np.clip(sim_log, -max_ref_threshold, max_ref_threshold).astype(np.float32)

    # Step 5: same-length pyramidinal smoothing（与真实数据管线一致）
    chr_pos_sim, x_smoothed = _running_mean_same_length_by_chromosome(
        sim_log, sim_var, window_size=window_size,
    )

    # Step 6: per-cell median 中心化
    x_smoothed = x_smoothed - np.median(x_smoothed, axis=1, keepdims=True)

    # Step 7: 第二次参考减法（**仍在 gene 维**）
    x_smoothed = _subtract_ref_bounds_matrix(x_smoothed, is_norm)

    # Step 8: 2^x → copy-ratio 空间（与真实数据管线一致）
    if output_space == "copy_ratio":
        x_final = np.power(2.0, x_smoothed, dtype=np.float32)
        neutral_anchor = 1.0
    else:
        x_final = x_smoothed.astype(np.float32)
        neutral_anchor = 0.0

    # ── 从各 CNV 染色体提取 emission 参数 ──────────────────────────────────
    n_total = x_final.shape[1]
    chr_starts = list(chr_pos_sim.values())
    chr_names  = list(chr_pos_sim.keys())
    chr_ends   = chr_starts[1:] + [n_total]
    chr_ranges = {c: (s, e) for c, s, e in zip(chr_names, chr_starts, chr_ends)}

    ratio_to_chr = {r: c for c, r, _ in chr_struct if r != 1.0}
    neutral_chr_list = [c for c, r, _ in chr_struct if r == 1.0]

    emit_means_list: list[float] = []
    emit_stds_list:  list[float] = []
    vals_by_state: list[np.ndarray] = []
    tumor_mask = ~is_norm

    for ratio in cnv_ratios:
        if ratio == 1.0:
            vals_list = []
            # R .get_gene_expr_by_cnv() 按 cnv key 合并所有 cnv:1 染色体，
            # 包括 chrA..chrE 以及可变长度 chr_F；不能只抽前两个 neutral chr。
            for cn in neutral_chr_list:
                if cn in chr_ranges:
                    ws, we = chr_ranges[cn]
                    vals_list.append(x_final[tumor_mask, ws:we].ravel())
            vals = np.concatenate(vals_list) if vals_list else x_final[tumor_mask].ravel()
        else:
            cn = ratio_to_chr.get(ratio)
            if cn is None or cn not in chr_ranges:
                closest = min(ratio_to_chr.keys(), key=lambda r: abs(r - ratio))
                cn = ratio_to_chr[closest]
            ws, we = chr_ranges[cn]
            vals = x_final[tumor_mask, ws:we].ravel()

        if len(vals) == 0:
            vals = np.full(10, neutral_anchor)
        vals = np.asarray(vals, dtype=np.float64)
        vals_by_state.append(vals)
        emit_means_list.append(float(np.mean(vals)))
        emit_stds_list.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 1e-5)
        logging.info(  # type: ignore
            f"  hspike ratio={ratio}: chr={ratio_to_chr.get(ratio, 'neutral')}, "
            f"n_vals={len(vals)}, mean={emit_means_list[-1]:.5f}, "
            f"sd={emit_stds_list[-1]:.5f}"
        )

    emit_means = np.array(emit_means_list, dtype=np.float64)
    emit_stds  = np.maximum(np.array(emit_stds_list, dtype=np.float64), 1e-5)

    sd_intercepts = np.zeros_like(emit_stds)
    sd_slopes = np.zeros_like(emit_stds)
    if return_sd_trend:
        nrounds = 100
        num_cells_grid = np.arange(1, 101, dtype=np.float64)
        for state_idx, vals in enumerate(vals_by_state):
            sd_vals = np.full(num_cells_grid.shape, np.nan, dtype=np.float64)
            for grid_idx, ncells in enumerate(num_cells_grid.astype(int)):
                # R:
                # vals <- replicate(nrounds, sample(expr_vals, size=ncells, replace=TRUE))
                # means <- Matrix::rowMeans(vals)  # 注意是 rowMeans，不是 colMeans
                sampled = rng.choice(vals, size=(ncells, nrounds), replace=True)
                means = sampled.mean(axis=1)
                sd_vals[grid_idx] = np.std(means, ddof=1) if means.size > 1 else np.nan

            valid_fit = np.isfinite(sd_vals) & (sd_vals > 0)
            if valid_fit.sum() >= 2:
                slope, intercept = np.polyfit(
                    np.log(num_cells_grid[valid_fit]),
                    np.log(sd_vals[valid_fit]),
                    deg=1,
                )
                sd_intercepts[state_idx] = float(intercept)
                sd_slopes[state_idx] = float(slope)
            else:
                sd_intercepts[state_idx] = float(np.log(max(emit_stds[state_idx], 1e-5)))
                sd_slopes[state_idx] = -0.5

    logging.info(  # type: ignore
        f"compute_hspike_emission_params done [{output_space}]: "
        f"means={np.round(emit_means, 4)}, stds={np.round(emit_stds, 4)}"
    )
    if return_sd_trend:
        logging.info(  # type: ignore
            "compute_hspike_emission_params sd trend: "
            f"intercepts={np.round(sd_intercepts, 4)}, slopes={np.round(sd_slopes, 4)}"
        )
        return emit_means, emit_stds, sd_intercepts, sd_slopes
    return emit_means, emit_stds
