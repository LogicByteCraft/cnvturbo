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
    _natural_sort,
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
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
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
        排除的染色体（默认排除 chrX/chrY）。
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

    # ── Step 2：库容归一化 → log2(x+1) ───────────────────────────────────────
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

    # ── 构建基因坐标过滤掩码 ──────────────────────────────────────────────────
    var_mask_null = adata.var["chromosome"].isnull()
    var_mask_excl = (
        adata.var["chromosome"].isin(exclude_chromosomes)
        if exclude_chromosomes is not None
        else np.zeros(adata.n_vars, dtype=bool)
    )
    keep_mask = ~(var_mask_null | var_mask_excl)
    expr_filt = log_expr[:, keep_mask]
    var_filt = adata.var.loc[keep_mask, ["chromosome", "start", "end"]]
    logging.info(f"  Genes for smoothing: {keep_mask.sum()} / {adata.n_vars}")  # type: ignore

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
        adata.uns[key_added] = {
            "chr_pos": chr_pos,
            "is_gene_space": True,
            "is_copy_ratio": bool(apply_2x_transform),
        }
        return None
    else:
        return chr_pos, res


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
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    n_sim_cells: int = 100,
    n_genes_per_chr: int = 400,
    cnv_ratios: tuple[float, ...] = (0.01, 0.5, 1.0, 1.5, 2.0, 3.0),
    output_space: str = "copy_ratio",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
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

    ref_raw = raw_counts[ref_mask]               # (n_ref, n_genes)

    var_mask_excl = (
        adata.var["chromosome"].isin(list(exclude_chromosomes))
        if exclude_chromosomes is not None
        else pd.Series(False, index=adata.var_names)
    )
    keep_mask = ~(adata.var["chromosome"].isnull() | var_mask_excl)
    n_genes_total   = int(keep_mask.sum())

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

    # ── 从真实参考细胞带放回采样合成正常细胞和肿瘤细胞基底 ──────────────────
    n_ref_cells = ref_raw.shape[0]
    norm_cell_idx  = rng.integers(0, n_ref_cells, size=n_sim_cells)
    tumor_cell_idx = rng.integers(0, n_ref_cells, size=n_sim_cells)

    keep_indices   = np.where(keep_mask)[0]
    sampled_gene_pos = keep_indices[sampled_idx]

    sim_norm       = ref_raw[norm_cell_idx][:, sampled_gene_pos].astype(np.float64)
    sim_tumor_base = ref_raw[tumor_cell_idx][:, sampled_gene_pos].astype(np.float64)
    sim_tumor      = sim_tumor_base * gene_ratio_vec[None, :]

    all_sim   = np.vstack([sim_norm, sim_tumor])
    is_norm   = np.array([True] * n_sim_cells + [False] * n_sim_cells)

    # ── 完整 R 管线（8 步，与 infercnv_r_compat 一致）──────────────────────
    totals  = np.maximum(all_sim.sum(axis=1, keepdims=True), 1.0)
    sf      = max(float(np.median(all_sim.sum(axis=1))), 1.0)
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
    tumor_mask = ~is_norm

    for ratio in cnv_ratios:
        if ratio == 1.0:
            vals_list = []
            for cn in neutral_chr_list[:2]:
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
        emit_means_list.append(float(np.mean(vals)))
        emit_stds_list.append(float(np.std(vals)))
        logging.info(  # type: ignore
            f"  hspike ratio={ratio}: chr={ratio_to_chr.get(ratio, 'neutral')}, "
            f"n_vals={len(vals)}, mean={emit_means_list[-1]:.5f}, "
            f"sd={emit_stds_list[-1]:.5f}"
        )

    emit_means = np.array(emit_means_list, dtype=np.float64)
    emit_stds  = np.maximum(np.array(emit_stds_list, dtype=np.float64), 1e-5)

    logging.info(  # type: ignore
        f"compute_hspike_emission_params done [{output_space}]: "
        f"means={np.round(emit_means, 4)}, stds={np.round(emit_stds, 4)}"
    )
    return emit_means, emit_stds
