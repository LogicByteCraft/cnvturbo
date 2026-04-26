"""CNV 推断主模块：滑窗平滑 + 降噪。

与原版 API 完全兼容，新增 backend 参数。

性能改进：
  - CPU 路径：用 numpy.lib.stride_tricks.sliding_window_view 替换
    np.apply_along_axis + np.convolve（纯 Python 循环），带 Numba 并行加速可选。
  - GPU 路径：用 torch.nn.functional.conv1d 在 CUDA 上一次处理整个染色体矩阵。
  - 并发改为 ThreadPoolExecutor（numpy 释放 GIL），消除多进程 IPC 开销。
"""

import itertools
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm

from cnvturbo._util import _ensure_array
from cnvturbo.tl._backend import get_backend, get_n_jobs, has_numba, has_torch


# ── Numba 可选加速 ────────────────────────────────────────────────────────────

def _build_numba_convolve():
    """惰性编译 Numba 版本的逐行金字塔卷积。

    按行并行（prange），适合 CPU 多核服务器。
    返回函数对象；若 numba 不可用则返回 None。
    """
    try:
        import numba  # noqa: PLC0415

        @numba.njit(parallel=True, cache=True, fastmath=True)
        def _convolve_rows_numba(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            """逐行 'valid' 模式一维卷积（Numba 并行版）。

            Parameters
            ----------
            x
                (N_cells, N_genes) float32 密集矩阵。
            kernel
                (n,) 已归一化的金字塔核。

            Returns
            -------
            (N_cells, N_genes - n + 1) 结果矩阵。
            """
            N, G = x.shape
            n = kernel.shape[0]
            out_G = G - n + 1
            result = np.empty((N, out_G), dtype=np.float32)
            for i in numba.prange(N):
                for j in range(out_G):
                    val = np.float32(0.0)
                    for k in range(n):
                        val += x[i, j + k] * kernel[k]
                    result[i, j] = val
            return result

        return _convolve_rows_numba
    except ImportError:
        return None


_NUMBA_CONVOLVE = None  # 惰性初始化，首次调用时赋值


def _get_numba_convolve():
    global _NUMBA_CONVOLVE
    if _NUMBA_CONVOLVE is None and has_numba():
        _NUMBA_CONVOLVE = _build_numba_convolve()
    return _NUMBA_CONVOLVE


# ── 公开 API ──────────────────────────────────────────────────────────────────


def infercnv(
    adata: AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    reference: np.ndarray | None = None,
    lfc_clip: float = 3,
    window_size: int = 100,
    step: int = 10,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    chunksize: int = 5000,
    n_jobs: int | None = None,
    backend: str = "auto",
    inplace: bool = True,
    layer: str | None = None,
    key_added: str = "cnv",
    calculate_gene_values: bool = False,
) -> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    """按基因组位置滑窗平均推断拷贝数变异（CNV）。

    与原版 API 完全兼容，新增 ``backend`` 参数控制计算后端。

    Parameters
    ----------
    adata
        注释数据矩阵。
    reference_key
        ``adata.obs`` 中标注正常/肿瘤的列名。
    reference_cat
        ``adata.obs[reference_key]`` 中代表正常细胞的值（可多个）。
    reference
        直接提供参考表达矩阵，覆盖 reference_key/reference_cat。
    lfc_clip
        对数倍变截断阈值。
    window_size
        滑窗基因数。
    step
        每隔 n 个窗口计算一次（节省内存）。
    dynamic_threshold
        小于 ``dynamic_threshold * STDDEV`` 的信号置零；None 禁用。
    exclude_chromosomes
        要排除的染色体列表，默认排除性染色体。
    chunksize
        每次处理的细胞数（控制内存）。GPU 路径下可适当调大。
    n_jobs
        CPU 并行线程数，None 使用全部核心。GPU 路径下忽略此参数。
    backend
        'auto' 自动选择，'cuda' 强制 GPU，'cpu' 强制 CPU。
    inplace
        True 将结果写入 adata，False 返回结果元组。
    layer
        使用的 adata 层，None 使用 adata.X。
    key_added
        结果在 adata.obsm 和 adata.uns 中的键名。
    calculate_gene_values
        True 时计算并存储每个基因的 CNV 值（内存消耗显著增加）。

    Returns
    -------
    inplace=True 时返回 None；否则返回 (chr_pos, cnv_matrix, per_gene_matrix)。
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns) != set():
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`. "
        )

    resolved_backend = get_backend(backend)
    n_jobs_eff = get_n_jobs(n_jobs)
    logging.info(f"infercnv backend={resolved_backend}, n_jobs={n_jobs_eff}")  # type: ignore

    # 构建 var 掩码（排除无位置 / 指定染色体）
    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        logging.warning(f"Skipped {np.sum(var_mask)} genes because they don't have a genomic position annotated. ")  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)

    tmp_adata = adata[:, ~var_mask]
    ref_arr = _get_reference(adata, reference_key, reference_cat, reference, layer)[:, ~var_mask]
    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]
    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()
    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]

    # ── GPU 路径：整批处理，无 process_map 开销 ───────────────────────────────
    if resolved_backend == "cuda":
        chr_pos, res, per_gene_mtx = _infercnv_gpu(
            expr, var, ref_arr, lfc_clip, window_size, step,
            dynamic_threshold, calculate_gene_values, chunksize,
        )
        if calculate_gene_values:
            per_gene_df = pd.DataFrame(per_gene_mtx, index=adata.obs.index)
            per_gene_df = per_gene_df.reindex(columns=adata.var_names, fill_value=np.nan)
            per_gene_mtx = per_gene_df.values
    else:
        # ── CPU 路径：ThreadPoolExecutor（numpy 释放 GIL，无 IPC 开销）────────
        cell_chunks = [
            expr[i : i + chunksize, :]
            for i in range(0, adata.shape[0], chunksize)
        ]

        results = [None] * len(cell_chunks)

        def _process(args):
            idx, chunk = args
            return idx, _infercnv_chunk(
                chunk, var, ref_arr, lfc_clip, window_size, step,
                dynamic_threshold, calculate_gene_values,
            )

        with ThreadPoolExecutor(max_workers=n_jobs_eff) as pool:
            futures = {pool.submit(_process, (i, c)): i for i, c in enumerate(cell_chunks)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="infercnv chunks"):
                idx, result = fut.result()
                results[idx] = result

        chr_pos_list, chunks, convolved_dfs = zip(*results, strict=False)
        res = scipy.sparse.vstack(chunks)
        chr_pos = chr_pos_list[0]

        if calculate_gene_values:
            per_gene_df = pd.concat(convolved_dfs, axis=0)
            per_gene_df.index = adata.obs.index
            per_gene_df = per_gene_df.reindex(columns=adata.var_names, fill_value=np.nan)
            per_gene_mtx = per_gene_df.values
        else:
            per_gene_mtx = None

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos}
        if calculate_gene_values:
            adata.layers[f"gene_values_{key_added}"] = per_gene_mtx
    else:
        return chr_pos, res, per_gene_mtx


# ── 内部实现：CPU 路径 ────────────────────────────────────────────────────────


def _natural_sort(l: Sequence):
    """自然排序（不依赖第三方库）。"""

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def _ordered_chromosomes(var: pd.DataFrame) -> list[str]:
    """返回平滑使用的染色体顺序，优先遵循 GTF 注释保留的 category 顺序。"""
    chrom = var["chromosome"]
    present = [str(x) for x in chrom.dropna().unique()]
    present_set = set(present)
    if isinstance(chrom.dtype, pd.CategoricalDtype) and chrom.cat.ordered:
        chromosomes = [str(c) for c in chrom.cat.categories if str(c) in present_set]
        seen = set(chromosomes)
        chromosomes.extend([c for c in present if c not in seen])
    else:
        chromosomes = _natural_sort(present)
    return [x for x in chromosomes if x.startswith("chr") and x != "chrM"]


def _build_pyramid_kernel(n: int) -> np.ndarray:
    """构造归一化金字塔核（与原版等价）。"""
    r = np.arange(1, n + 1, dtype=np.float32)
    pyramid = np.minimum(r, r[::-1])
    return (pyramid / pyramid.sum()).astype(np.float32)


def _running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """按行执行金字塔加权滑动均值（'valid' 模式，按步长降采样）。

    性能改进：用 sliding_window_view + 矩阵乘法替换逐行 np.convolve，
    在 numba 可用时进一步提升到全并行。

    Parameters
    ----------
    x
        (N_cells, N_genes) 矩阵（密集或稀疏）。
    n
        滑窗大小（基因数）。
    step
        每 step 个窗口输出一个值。
    gene_list
        基因名列表（仅 calculate_gene_values=True 时使用）。
    calculate_gene_values
        是否计算并返回每基因 CNV 均值。

    Returns
    -------
    (smoothed_x, convolved_gene_values)
        smoothed_x: (N_cells, n_windows // step)
        convolved_gene_values: DataFrame 或 None
    """
    if scipy.sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)

    N, G = x.shape

    if n >= G:
        # 基因数不足一个窗口：退化为对整行平均
        n = G
        smoothed_x = x.mean(axis=1, keepdims=True).astype(np.float32)
        if calculate_gene_values:
            convolved_gene_values = pd.DataFrame(
                np.repeat(smoothed_x, len(gene_list), axis=1), columns=gene_list
            )
        else:
            convolved_gene_values = None
        return smoothed_x, convolved_gene_values

    kernel = _build_pyramid_kernel(n)  # (n,) 已归一化

    # 尝试 Numba 加速（首次调用会触发 JIT 编译）
    nb_fn = _get_numba_convolve()
    if nb_fn is not None:
        smoothed_full = nb_fn(x, kernel)  # (N, G-n+1)
    else:
        # 纯 numpy 向量化：sliding_window_view 避免 Python 级循环
        from numpy.lib.stride_tricks import sliding_window_view  # noqa: PLC0415

        windows = sliding_window_view(x, n, axis=1)  # (N, G-n+1, n)
        smoothed_full = (windows * kernel).sum(axis=-1).astype(np.float32)

    # 步长降采样
    smoothed_x = smoothed_full[:, ::step]  # (N, n_windows)

    if calculate_gene_values:
        # 保留对应窗口的中心基因名，用于后续 per-gene 计算
        convolution_indices = np.arange(0, smoothed_full.shape[1], step)
        center_genes = gene_list[convolution_indices + n // 2] if gene_list is not None else None
        convolved_gene_values = _calculate_gene_averages(
            gene_list[np.array([np.arange(i, i + n) for i in convolution_indices])],
            smoothed_x,
        )
    else:
        convolved_gene_values = None

    return smoothed_x, convolved_gene_values


def _calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    """计算每个基因在其所属窗口中的平均 CNV 值。"""
    gene_to_values: dict = {}
    length = len(convolved_gene_names[0])
    flatten_list = list(convolved_gene_names.flatten())

    for sample, row in enumerate(smoothed_x):
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    return pd.DataFrame(gene_to_values).T


def get_convolution_indices(x, n):
    """返回每个窗口覆盖的基因索引数组。"""
    return np.array([np.arange(i, i + n) for i in range(x.shape[1] - n + 1)])


# ── R smooth_helper 端点修正算法（精确复现） ────────────────────────────────


def _running_mean_same_length(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 101,
) -> np.ndarray:
    """R smooth_by_chromosome 等价实现：输出长度 = 输入基因数。

    精确复现 R inferCNV_ops.R 中 .smooth_helper / .smooth_center_helper：
      - 内部窗口（centered, 长度 ≥ window_length）：与 R `filter(...,sides=2)` 等价
        custom_filter_numerator = c(1..tail, tail+1, tail..1)
        custom_filter_denominator = ((window-1)/2)^2 + window
      - 端点窗口（不足 tail_length 时）：用动态分母与对应数值修正
        denominator = ((window-1)/2)^2 + window - r_left*(r_left+1)/2 - r_right*(r_right+1)/2

    Parameters
    ----------
    x : (N_cells, N_genes) 输入矩阵
    n : 窗口大小（必须为奇数，与 R window_length 一致）

    Returns
    -------
    smoothed : (N_cells, N_genes) 与输入同形状（每基因有一个平滑值）
    """
    if scipy.sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)

    N, G = x.shape

    if n < 2:
        return x.astype(np.float32)

    if G < 2:
        return x.astype(np.float32)

    # 取奇数化窗口
    if n % 2 == 0:
        n -= 1
    tail_length = (n - 1) // 2  # R: tail_length = (window_length - 1) / 2

    # ── 内部窗口卷积（与 R filter(sides=2) 等价） ────────────────────────
    # numerator = c(1..tail, tail+1, tail..1)
    # denominator = ((window-1)/2)^2 + window  = tail^2 + n
    custom_filter_denom = float(tail_length * tail_length + n)
    custom_filter_num = np.concatenate([
        np.arange(1, tail_length + 1, dtype=np.float64),
        [tail_length + 1],
        np.arange(tail_length, 0, -1, dtype=np.float64),
    ])
    kernel_normed = (custom_filter_num / custom_filter_denom).astype(np.float64)  # (n,)

    # 中心区域（长度 G-n+1）：valid 卷积；写到输出 [tail_length : tail_length + (G-n+1)]
    out = x.copy()  # 默认返回原值（端点未平滑前的原始值）

    if G >= n:
        # 用 sliding_window_view + 矩阵乘法
        from numpy.lib.stride_tricks import sliding_window_view  # noqa: PLC0415
        windows = sliding_window_view(x, n, axis=1)            # (N, G-n+1, n)
        center = (windows * kernel_normed).sum(axis=-1)        # (N, G-n+1)
        out[:, tail_length: tail_length + (G - n + 1)] = center

    # ── 端点修正（R 的 .smooth_helper 端点循环）─────────────────────────
    # 对每个 tail_end ∈ [0, iteration_range) 修正左右端点
    # iteration_range = tail_length if G > n else ceil(G/2)
    iteration_range = tail_length if G > n else (G + 1) // 2
    obs_count = G

    # numerator_counts_vector: 同 custom_filter_num（R 中 size = window_length）
    num_vec = custom_filter_num  # (n,)

    for tail_end_0 in range(iteration_range):
        # R: tail_end ∈ [1..iteration_range]，转为 0-indexed
        tail_end = tail_end_0  # 左端 0-indexed
        end_tail = obs_count - tail_end_0 - 1  # 右端 0-indexed（R: obs_count - tail_end + 1）

        d_left = tail_end_0  # tail_end - 1（R 是 1-based，差 1）
        d_right = obs_count - tail_end_0 - 1
        d_right = min(d_right, tail_length)

        r_left = tail_length - d_left
        r_right = tail_length - d_right

        denom = float(
            tail_length * tail_length + n
            - (r_left * (r_left + 1)) / 2
            - (r_right * (r_right + 1)) / 2
        )
        if denom <= 0:
            continue

        # 取 numerator_range 段（R: num_vec[(tail_length+1-d_left):(tail_length+1+d_right)]）
        num_range = num_vec[(tail_length - d_left): (tail_length + d_right + 1)]  # 长度 = d_left + d_right + 1
        # 左端：obs_data[seq_len(tail_end + d_right)]，R 1-based [1:tail_end+d_right]
        # 0-based: [0 : tail_end_0 + d_right + 1] → 长度 tail_end_0 + d_right + 1 = d_left + d_right + 1 ✓
        left_chunk_end = tail_end_0 + d_right + 1
        left_chunk = x[:, :left_chunk_end]   # (N, L)

        # 右端：obs_data[(end_tail - d_right):obs_length]，R 1-based 终点为 obs_length
        # 0-based: [end_tail - d_right : obs_count] → 长度 d_right + tail_end_0 + 1 = d_left + d_right + 1 ✓
        right_chunk_start = end_tail - d_right
        right_chunk = x[:, right_chunk_start: obs_count]

        # 应用：sum(left * num_range) / denom；右端用 reverse(num_range)
        out[:, tail_end] = (left_chunk * num_range).sum(axis=1) / denom
        out[:, end_tail] = (right_chunk * num_range[::-1]).sum(axis=1) / denom

    return out.astype(np.float32)


def _running_mean_same_length_by_chromosome(
    expr: np.ndarray,
    var: pd.DataFrame,
    window_size: int = 101,
) -> tuple[dict, np.ndarray]:
    """R smooth_by_chromosome 等价：按染色体独立平滑，保持基因维度长度不变。

    Returns
    -------
    chr_pos : {chr_name: gene_start_index}（基因级染色体边界）
    smoothed_genes : (N_cells, N_genes_filt) 平滑后矩阵（每基因一列）
    """
    chromosomes = _ordered_chromosomes(var)

    chunks = []
    chr_pos = {}
    cum = 0
    for chr in chromosomes:
        gene_idx = var.index.get_indexer(
            var.loc[var["chromosome"] == chr]
            .sort_values("start", kind="mergesort")
            .index.values
        )
        if len(gene_idx) == 0:
            continue
        chr_pos[chr] = cum
        sub = expr[:, gene_idx]
        chunks.append(_running_mean_same_length(sub, n=window_size))
        cum += len(gene_idx)

    return chr_pos, np.hstack(chunks)


def _running_mean_by_chromosome(
    expr, var, window_size, step, calculate_gene_values
) -> tuple[dict, np.ndarray, pd.DataFrame | None]:
    """按染色体独立计算滑动均值并拼接。"""
    chromosomes = _ordered_chromosomes(var)

    running_means = [
        _running_mean_for_chromosome(chr, expr, var, window_size, step, calculate_gene_values)
        for chr in chromosomes
    ]

    running_means, convolved_dfs = zip(*running_means, strict=False)

    chr_start_pos = {}
    for chr, i in zip(chromosomes, np.cumsum([0] + [x.shape[1] for x in running_means]), strict=False):
        chr_start_pos[chr] = i

    if calculate_gene_values:
        convolved_dfs = pd.concat(convolved_dfs, axis=1)

    return chr_start_pos, np.hstack(running_means), convolved_dfs


def _running_mean_for_chromosome(chr, expr, var, window_size, step, calculate_gene_values):
    """单条染色体的滑动均值。"""
    genes = (
        var.loc[var["chromosome"] == chr]
        .sort_values("start", kind="mergesort")
        .index.values
    )
    tmp_x = expr[:, var.index.get_indexer(genes)]
    x_conv, convolved_gene_values = _running_mean(
        tmp_x, n=window_size, step=step,
        gene_list=genes, calculate_gene_values=calculate_gene_values,
    )
    return x_conv, convolved_gene_values


def _get_reference(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
    layer: str | None,
) -> np.ndarray:
    """提取参考基因表达矩阵（支持多参考类别的 bounded 模式）。"""
    X = adata.layers[layer] if layer is not None else adata.X

    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                "Using mean of all cells as reference. For better results, "
                "provide either `reference`, or both `reference_key` and `reference_cat`. "
            )  # type: ignore
            reference = np.mean(X, axis=0)
        else:
            obs_col = adata.obs[reference_key]
            if isinstance(reference_cat, str):
                reference_cat = [reference_cat]
            reference_cat = np.array(reference_cat)
            reference_cat_in_obs = np.isin(reference_cat, obs_col)
            if not np.all(reference_cat_in_obs):
                raise ValueError(
                    "The following reference categories were not found in "
                    f"adata.obs[reference_key]: {reference_cat[~reference_cat_in_obs]}"
                )
            reference = np.vstack(
                [np.mean(X[obs_col.values == cat, :], axis=0) for cat in reference_cat]
            )

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError("Reference must match the number of genes in AnnData. ")

    return reference


def _infercnv_chunk(
    tmp_x, var, reference, lfc_cap, window_size, step, dynamic_threshold,
    calculate_gene_values=False, skip_cell_median=False,
):
    """对一批细胞执行 5 步 CNV 推断。

    Step 1: 参考中心化（LFC）
    Step 2: Clip
    Step 3: 按染色体滑窗平滑
    Step 4: 按细胞中位数中心化（skip_cell_median=True 时跳过，用于 R-compat 管线）
    Step 5: 动态阈值噪声过滤

    Parameters
    ----------
    skip_cell_median
        若 True，跳过 Step 4 的 per-cell 中位数中心化。
        在 infercnv_r_compat() 中使用：per-gene 参考减法已正确中心化，
        再做 per-cell 中心化会把染色体级 CNV 信号消除。
    """
    # Step 1：中心化
    if reference.shape[0] == 1:
        x_centered = tmp_x - reference[0, :]
    else:
        ref_min = np.min(reference, axis=0)
        ref_max = np.max(reference, axis=0)
        x_centered = np.zeros(tmp_x.shape, dtype=tmp_x.dtype)
        above_max = tmp_x > ref_max
        below_min = tmp_x < ref_min
        x_centered[above_max] = _ensure_array(tmp_x - ref_max)[above_max]
        x_centered[below_min] = _ensure_array(tmp_x - ref_min)[below_min]

    x_centered = _ensure_array(x_centered)
    # Step 2：截断
    x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap)
    # Step 3：滑窗平滑
    chr_pos, x_smoothed, conv_df = _running_mean_by_chromosome(
        x_clipped, var, window_size=window_size, step=step,
        calculate_gene_values=calculate_gene_values,
    )
    # Step 4：按细胞中位数中心化
    # 注意：infercnv_r_compat() 已在 Step 4 完成 per-gene 参考减法，
    # 此处的 per-cell 中心化会消除染色体级 CNV 偏移，必须跳过。
    if skip_cell_median:
        x_res = x_smoothed
        gene_res = conv_df if calculate_gene_values else None
    else:
        x_res = x_smoothed - np.median(x_smoothed, axis=1)[:, np.newaxis]
        gene_res = (conv_df - np.median(conv_df, axis=1)[:, np.newaxis]) if calculate_gene_values else None

    # Step 5：噪声过滤
    if dynamic_threshold is not None:
        noise_thres = dynamic_threshold * np.std(x_res)
        x_res[np.abs(x_res) < noise_thres] = 0
        if calculate_gene_values:
            gene_res[np.abs(gene_res) < noise_thres] = 0

    return chr_pos, scipy.sparse.csr_matrix(x_res), gene_res


# ── 内部实现：GPU 路径 ────────────────────────────────────────────────────────


def _infercnv_gpu(
    expr, var, reference, lfc_cap, window_size, step,
    dynamic_threshold, calculate_gene_values, chunksize,
):
    """GPU 加速版 infercnv（需要 torch + CUDA）。

    逐染色体在 GPU 上执行卷积；细胞维度按 chunksize 分批以适配 GPU 显存。
    返回与 CPU 路径相同的 (chr_pos, sparse_cnv_matrix, per_gene_matrix)。
    """
    import torch  # noqa: PLC0415

    device = torch.device("cuda")
    chromosomes = _ordered_chromosomes(var)

    n_cells = expr.shape[0]
    all_results = []      # 收集每个 chunk 的 CNV 矩阵片段
    chr_pos = None        # 所有 chunk 共享相同 chr_pos

    for chunk_start in range(0, n_cells, chunksize):
        chunk_end = min(chunk_start + chunksize, n_cells)
        tmp_x = expr[chunk_start:chunk_end, :]
        if scipy.sparse.issparse(tmp_x):
            tmp_x = tmp_x.toarray()
        tmp_x = tmp_x.astype(np.float32)

        # Step 1：参考中心化
        if reference.shape[0] == 1:
            x_centered = tmp_x - reference[0, :]
        else:
            ref_min = np.min(reference, axis=0)
            ref_max = np.max(reference, axis=0)
            x_centered = np.zeros_like(tmp_x)
            above_max = tmp_x > ref_max
            below_min = tmp_x < ref_min
            x_centered[above_max] = (tmp_x - ref_max)[above_max]
            x_centered[below_min] = (tmp_x - ref_min)[below_min]

        # Step 2：截断
        x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap).astype(np.float32)

        # Step 3：GPU 滑窗平滑（逐染色体）
        chr_smoothed_list = []
        chr_pos_local = {}
        col_offset = 0

        for chrom in chromosomes:
            genes = (
                var.loc[var["chromosome"] == chrom]
                .sort_values("start", kind="mergesort")
                .index.values
            )
            col_idx = var.index.get_indexer(genes)
            chrom_x = x_clipped[:, col_idx]  # (N_chunk, N_chrom_genes)
            n_genes = chrom_x.shape[1]
            n_win = window_size if window_size < n_genes else n_genes

            kernel = _build_pyramid_kernel(n_win)  # (n_win,)

            # 传入 GPU
            x_gpu = torch.from_numpy(chrom_x).to(device)          # (N, G)
            k_gpu = torch.from_numpy(kernel).to(device)             # (n_win,)

            # conv1d 需要 (batch, channels, width)；逐行卷积用 groups=N
            # 方法：reshape to (N, 1, G)，用 (1, 1, n_win) 核，padding=0 → 'valid'
            x_3d = x_gpu.unsqueeze(1)                               # (N, 1, G)
            k_3d = k_gpu.unsqueeze(0).unsqueeze(0)                 # (1, 1, n_win)
            smoothed_gpu = torch.nn.functional.conv1d(x_3d, k_3d, padding=0)  # (N, 1, G-n+1)
            smoothed_gpu = smoothed_gpu.squeeze(1)                  # (N, G-n+1)

            # 步长降采样
            smoothed_gpu = smoothed_gpu[:, ::step]                  # (N, n_windows)

            chr_pos_local[chrom] = col_offset
            col_offset += smoothed_gpu.shape[1]
            chr_smoothed_list.append(smoothed_gpu)

        if chr_pos is None:
            chr_pos = chr_pos_local

        # 拼接所有染色体
        x_smoothed_gpu = torch.cat(chr_smoothed_list, dim=1)       # (N, total_windows)

        # Step 4：按细胞中位数中心化（GPU）
        med = torch.median(x_smoothed_gpu, dim=1).values.unsqueeze(1)
        x_res_gpu = x_smoothed_gpu - med

        # Step 5：噪声过滤（GPU）
        if dynamic_threshold is not None:
            std_val = x_res_gpu.std()
            noise_thres = dynamic_threshold * std_val
            x_res_gpu[x_res_gpu.abs() < noise_thres] = 0.0

        # 转回 CPU → 稀疏
        x_res_cpu = x_res_gpu.cpu().numpy()
        all_results.append(scipy.sparse.csr_matrix(x_res_cpu))

    res = scipy.sparse.vstack(all_results)
    # calculate_gene_values 在 GPU 路径下暂不支持（罕用且内存消耗大）
    per_gene_mtx = None
    if calculate_gene_values:
        logging.warning(  # type: ignore
            "calculate_gene_values=True is not supported in GPU mode; "
            "falling back to None. Run with backend='cpu' if per-gene values are needed."
        )

    return chr_pos, res, per_gene_mtx
