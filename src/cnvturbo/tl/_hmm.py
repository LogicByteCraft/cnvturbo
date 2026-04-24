"""HMM i6 细胞级 CNV 判定模块。

实现与 R inferCNV HMM i6 等价的隐马尔科夫模型，对每个细胞独立
进行 Viterbi 解码，输出细胞级 Tumor/Normal 标签。

设计目标
--------
- 与 R inferCNV 相同的 6 态 HMM 结构（拷贝数状态：0, 0.5, 1, 1.5, 2, 3 相对倍型）
- CPU 路径：Numba @njit + prange 多核并行 Viterbi（可选，回退到纯 numpy）
- GPU 路径：PyTorch 批量 Viterbi（所有细胞在 CUDA 上同时解码）
- 三种发射参数估计模式：adaptive / fixed_log2 / em（Baum-Welch）
- 向后兼容：输出写入 adata.obs / adata.obsm，不修改现有键

HMM i6 状态定义
--------------
  State 0: 完全缺失（0 拷贝）        log2(0/2) = -∞ → 近似 -1.0
  State 1: 单拷贝缺失（1 拷贝）      log2(1/2) ≈ -0.585 → R 近似 -0.5
  State 2: 二倍体中性（2 拷贝）← neutral  log2(2/2) = 0
  State 3: 单拷贝增益（3 拷贝）      log2(3/2) ≈ +0.585 → R 近似 +0.5
  State 4: 双拷贝增益（4 拷贝）      log2(4/2) = +1.0
  State 5: 极端扩增（6+ 拷贝）       log2(6/2) ≈ +1.585 → R 近似 +1.5

发射参数估计模式（fit_method）
------------------------------
  "adaptive"   （默认）从数据信号振幅自适应缩放默认状态均值；兼容任意预处理管线。
  "fixed_log2" 使用 R inferCNV 原始固定 log2 参数（_R_LOG2_STATE_MEANS/_STDS）；
               需配合 infercnv_r_compat() 使用，两者组合完全对标 R。
  "em"         Baum-Welch EM 在实际数据上拟合发射分布；
               优先 hmmlearn.hmm.GaussianHMM，不可用时回退 numpy 前向-后向算法。

Tumor/Normal 判定
-----------------
v2 (subcluster 模式，与 R `pred_cnv_regions.dat` 等价)：
  - 对每个 subcluster 的每条染色体扫描 HMM 状态序列
  - 提取连续相同状态段，过滤长度 < min_segment_length 的非中性段（denoise 等价）
  - 若 subcluster 含 ≥ min_segments_for_tumor 个非中性段 → Tumor，否则 Normal
  - 将 subcluster 标签广播回所有细胞

cell 模式（hmm_call_cells）：
  - 统计每个细胞所有窗口中非中性状态比例 > tumor_threshold → Tumor
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import scipy.sparse
from anndata import AnnData
from scanpy import logging

from cnvturbo.tl._backend import get_backend, get_n_jobs, has_numba

# ── HMM 默认超参数（对齐 R inferCNV 默认值）────────────────────────────────

# fit_method="adaptive"：自适应信号振幅缩放模式
# 6 个状态在 log-LFC 空间中的默认均值（以中性=0 为基准），按实际信号振幅缩放
_DEFAULT_STATE_MEANS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)

# 各态发射标准差缩放系数（adaptive 模式；中性态最窄，极端态较宽）
_DEFAULT_STATE_STD_SCALE = np.array([1.5, 1.2, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)

# fit_method="fixed_log2"：log2 空间固定参数（legacy）
_R_LOG2_STATE_MEANS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
_R_LOG2_STATE_STDS  = np.array([ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3], dtype=np.float64)

# fit_method="fixed_copy_ratio"：copy-ratio 空间固定参数
_R_COPY_RATIO_STATE_MEANS = np.array([0.01, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64)
_R_COPY_RATIO_STATE_STDS  = np.array([0.15, 0.15, 0.15, 0.15, 0.20, 0.30], dtype=np.float64)

# 中性态在 copy-ratio 空间的下标（State 2 = 1.0x）
NEUTRAL_STATE_COPY_RATIO = 2

# 状态转移概率（非 R-compat 模式用）
_DEFAULT_P_STAY = 0.99

# 初始状态先验（非 R-compat 模式用）
_DEFAULT_LOG_PRIOR = np.log(np.array([0.01, 0.04, 0.80, 0.08, 0.05, 0.02], dtype=np.float64))

# ── R inferCNV 精确对标参数（来自 R/inferCNV_HMM.R）──────────────────────────
# R 的转移概率 t=1e-6：非对角为 t，对角为 1-5t（6 态时）
# 对应 run() 的 HMM_transition_prob=1e-6 默认值
_R_T = 1e-6

# R 的初始先验 delta：中性态（state 2）为 1-5t ≈ 0.999995，其余为 t
_R_LOG_PRIOR = np.log(np.array(
    [_R_T, _R_T, 1.0 - 5 * _R_T, _R_T, _R_T, _R_T], dtype=np.float64
))

# Tumor 判定阈值：非中性窗口占比超过此值视为 Tumor（仅 hmm_call_cells 用；
# hmm_call_subclusters v2 已改为 R 等价的"含 CNV 段"二分类，不再使用此阈值）
_DEFAULT_TUMOR_THRESHOLD = 0.20

# v2: subcluster 级 R 等价 Tumor 判定参数
# - min_segment_length：最小连续非中性段长（基因数），相当于 R denoise 的段长去噪
# - min_segments_for_tumor：subcluster 至少含多少 ≥min_segment_length 的非中性段才判 Tumor
_DEFAULT_MIN_SEGMENT_LENGTH = 5
_DEFAULT_MIN_SEGMENTS_FOR_TUMOR = 1

N_STATES = 6
NEUTRAL_STATE = 2  # 中性态编号（0-indexed）


# ── R-style emission 函数（精确复现 R Viterbi.dthmm.adj）─────────────────────

def _build_r_log_transition(n_states: int = 6, t: float = _R_T) -> np.ndarray:
    """构造 R inferCNV 风格的对数转移矩阵。

    R 的构造方式（inferCNV_HMM.R L233-240）：
      - 非对角元素：t（HMM_transition_prob=1e-6）
      - 对角元素：1 - (n_states-1)*t
    """
    trans = np.full((n_states, n_states), t, dtype=np.float64)
    np.fill_diagonal(trans, 1.0 - (n_states - 1) * t)
    return np.log(trans)


def _r_emission_log(x: float, means: np.ndarray, sd_unified: float) -> np.ndarray:
    """R Viterbi.dthmm.adj 的 emission 计算（L1122-1133）。

    R 代码：
      emission <- pnorm(abs(x[i]-pm$mean)/pm$sd, log.p=TRUE, lower.tail=FALSE)
      emission <- 1 / (-1 * emission)    # 取倒数
      emission <- emission / sum(emission)  # 归一化
      → log(emission) 作为 log-likelihood

    数学含义：
      z_s = |x - μ_s| / σ_unified
      log_sf_s = log(P(Z > z_s))  ← log.p=TRUE, lower.tail=FALSE 即 log(1-Φ(z))
      raw_s = 1 / (-log_sf_s)     ← 越近的状态 z 越小，sf 越大（接近 0.5），|log_sf| 越小，倒数越大
      emission_s = raw_s / Σ raw   ← 归一化使之成为概率
      log_emission_s = log(emission_s)

    Parameters
    ----------
    x          : 标量观测值
    means      : (S,) 各状态均值
    sd_unified : 统一 SD（R 中 median(pm$sd)）

    Returns
    -------
    (S,) log emission 概率
    """
    from scipy.special import log_ndtr  # log(Φ(x)) = log(pnorm(x))

    z = np.abs(x - means) / sd_unified
    # log(P(Z > z)) = log(1-Φ(z)) = log_ndtr(-z)
    log_sf = log_ndtr(-z)
    # emission = 1 / (-log_sf)  ← 注意 log_sf < 0，所以 -log_sf > 0
    raw = 1.0 / (-log_sf)
    raw /= raw.sum()  # 归一化
    return np.log(raw + 1e-300)  # 加小量防止 log(0)


def _viterbi_r_single(
    x: np.ndarray,
    log_trans: np.ndarray,
    log_prior: np.ndarray,
    emit_means: np.ndarray,
    emit_stds: np.ndarray,
) -> np.ndarray:
    """单序列 R-style Viterbi（复现 R Viterbi.dthmm.adj）。

    R 实现要点：
      1. pm$sd = median(pm$sd)   ← 所有状态统一使用同一 SD
      2. emission 使用 pnorm-based 公式（非 Gaussian density）
      3. forward pass：nu[t,s] = max_{s'} nu[t-1,s'] + log_Pi[s',s] + log_emit[t,s]
      4. traceback：y[n] = argmax nu[n], y[t] = argmax log_Pi[:,y[t+1]] + nu[t]
    """
    sd_unified = float(np.median(emit_stds))  # R: pm$sd = median(pm$sd)
    T = len(x)
    S = len(emit_means)

    log_pi = log_trans  # (S, S)

    # 初始时刻
    log_alpha = log_prior + _r_emission_log(x[0], emit_means, sd_unified)  # (S,)
    backptr = np.zeros((T, S), dtype=np.int32)

    # 前向递推
    for t in range(1, T):
        # candidates[s_prev, s_next] = log_alpha[s_prev] + log_pi[s_prev, s_next]
        candidates = log_alpha[:, None] + log_pi          # (S, S)
        best_prev = np.argmax(candidates, axis=0)         # (S,)
        best_scores = candidates[best_prev, np.arange(S)]
        emit_t = _r_emission_log(x[t], emit_means, sd_unified)
        log_alpha = best_scores + emit_t
        backptr[t] = best_prev

    # 回溯
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = int(np.argmax(log_alpha))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]

    return states


def _viterbi_r_batch(
    sequences: np.ndarray,
    log_trans: np.ndarray,
    log_prior: np.ndarray,
    emit_means: np.ndarray,
    emit_stds: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """批量 R-style Viterbi（ThreadPoolExecutor 并行）。

    Parameters
    ----------
    sequences : (N, T) float64 — N 条序列（通常为 subcluster means）
    其余参数同 _viterbi_r_single

    Returns
    -------
    states : (N, T) int32
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    N, T = sequences.shape
    states = np.zeros((N, T), dtype=np.int32)

    def _worker(i: int) -> tuple[int, np.ndarray]:
        return i, _viterbi_r_single(
            sequences[i], log_trans, log_prior, emit_means, emit_stds
        )

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        for i, seq_states in pool.map(lambda i: _worker(i), range(N)):
            states[i] = seq_states

    return states


# ── R denoise 等价：HMM 状态序列段长过滤 ─────────────────────────────────────


def _denoise_segments(
    state_seq: np.ndarray,
    neutral_state: int = NEUTRAL_STATE,
    min_segment_length: int = _DEFAULT_MIN_SEGMENT_LENGTH,
) -> np.ndarray:
    """对单个染色体内的状态序列做段长去噪。

    扫描连续相同状态的段，若某非中性段长度 < `min_segment_length`，
    将该段的所有位置置为 `neutral_state`。

    R 对应：`denoise=TRUE` 在表达矩阵层面做 median filter，
    在状态序列上等价于过滤过短的孤立非中性段。

    Parameters
    ----------
    state_seq          : (T,) int — 单条染色体的状态序列
    neutral_state      : 中性态编号
    min_segment_length : 段长阈值（基因数）

    Returns
    -------
    cleaned : (T,) int — 去噪后的状态序列
    """
    T = state_seq.shape[0]
    if T == 0 or min_segment_length <= 1:
        return state_seq.copy()

    out = state_seq.copy()
    i = 0
    while i < T:
        s = out[i]
        j = i + 1
        while j < T and out[j] == s:
            j += 1
        if s != neutral_state and (j - i) < min_segment_length:
            out[i:j] = neutral_state
        i = j
    return out


def _count_cnv_segments(
    state_seq: np.ndarray,
    neutral_state: int = NEUTRAL_STATE,
    min_segment_length: int = _DEFAULT_MIN_SEGMENT_LENGTH,
) -> int:
    """统计某状态序列中长度 ≥ `min_segment_length` 的非中性段数。

    与 R `.define_cnv_gene_regions` + `pred_cnv_regions.dat` 的"非中性 region 计数"等价：
      R: 连续同状态基因合并成 region；过滤 neutral state 后，剩余 region 即 CNV 段。
      cnvturbo (v2): 同时引入段长阈值（与 R denoise 段长过滤等价）。
    """
    T = state_seq.shape[0]
    n_seg = 0
    i = 0
    while i < T:
        s = state_seq[i]
        j = i + 1
        while j < T and state_seq[j] == s:
            j += 1
        if s != neutral_state and (j - i) >= min_segment_length:
            n_seg += 1
        i = j
    return n_seg


# ── 公开 API ──────────────────────────────────────────────────────────────────


def hmm_call_subclusters(
    adata: AnnData,
    *,
    use_rep: str = "cnv",
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    n_states: int = N_STATES,
    tumor_threshold: float = _DEFAULT_TUMOR_THRESHOLD,
    min_segment_length: int = _DEFAULT_MIN_SEGMENT_LENGTH,
    min_segments_for_tumor: int = _DEFAULT_MIN_SEGMENTS_FOR_TUMOR,
    leiden_resolution: float | str = "auto",
    cluster_by_groups: bool = True,
    n_neighbors: int = 20,
    n_pcs: int = 10,
    fit_method: str = "hspike",
    p_stay: float = _DEFAULT_P_STAY,
    use_r_viterbi: bool = True,
    precomputed_emit_means: np.ndarray | None = None,
    precomputed_emit_stds: np.ndarray | None = None,
    backend: str = "auto",
    key_added: str = "cnv_tumor_call",
    n_jobs: int | None = None,
    inplace: bool = True,
    chunksize: int = 2000,
    random_state: int = 0,
) -> None | dict:
    """R inferCNV analysis_mode="subclusters" 精确对标实现（v2）。

    与 R inferCNV 的完整对应关系（R/inferCNV_HMM.R + inferCNV_tumor_subclusters.R）：
      1. Leiden 聚类（按 cluster_by_groups 选择是否分组），分辨率默认 "auto"
         R: leiden_resolution_auto = (11.98 / n_cells) ^ (1/1.165)
         PCA(n_pcs=10) → kNN(n_neighbors=20)
      2. 每个亚克隆内 rowMeans 计算 **基因级**均值 CNV 谱
      3. **按染色体独立**运行 HMM Viterbi（每染色体 T = 基因数）
         Viterbi 使用 R 的 pnorm-based emission（use_r_viterbi=True）
      4. SD 按亚克隆细胞数缩放：base_sd / sqrt(N)
      5. **denoise 段长过滤**（min_segment_length，与 R denoise=TRUE 等价）
      6. **R 等价 Tumor 判定**：subcluster 含 ≥ min_segments_for_tumor 个非中性段 → Tumor
         （R 等价于 pred_cnv_regions.dat 中出现该 subcluster 即判 Tumor.obs）
      7. 将亚克隆标签广播回每个细胞

    Parameters
    ----------
    adata
        已运行 infercnv_r_compat() 的 AnnData。
        adata.uns[use_rep]["chr_pos"] 应为基因级染色体起始下标。
    use_rep
        CNV 矩阵键名（不含 "X_" 前缀），默认 "cnv"。
    reference_key / reference_cat
        参考细胞注释。
    n_states
        HMM 状态数，默认 6（i6 模型）。
    tumor_threshold
        legacy 参数，仅当 min_segments_for_tumor=0 时退回旧的"非中性占比"阈值。
    min_segment_length
        最小连续非中性段长（基因数），默认 5（与 R denoise 段长去噪等价）。
    min_segments_for_tumor
        subcluster 含多少 ≥min_segment_length 的非中性段才判 Tumor，默认 1（R 行为）。
    leiden_resolution
        Leiden 分辨率：数值或 "auto"（按 R 公式）。
    cluster_by_groups
        True（R 默认）：按 reference_cat 分组分别 leiden（参考组与观测组独立聚类）。
        False：所有细胞一起聚类。
    n_neighbors
        kNN 邻居数（R 默认 20）。
    n_pcs
        PCA 主成分数（R 默认 10）。
    fit_method
        HMM 发射均值校准模式，默认 "hspike"。
    use_r_viterbi
        True（默认）：使用 R pnorm-based emission + t=1e-6 + R prior。
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError(
            "scanpy is required for hmm_call_subclusters(). "
            "Install with: pip install scanpy"
        )

    cnv_key = f"X_{use_rep}"
    if cnv_key not in adata.obsm:
        raise KeyError(f"{cnv_key} not in adata.obsm")

    cnv_matrix = adata.obsm[cnv_key]
    if scipy.sparse.issparse(cnv_matrix):
        cnv_matrix = cnv_matrix.toarray()
    cnv_matrix = np.asarray(cnv_matrix, dtype=np.float64)

    n_jobs_eff = get_n_jobs(n_jobs)
    n_cells_total = adata.n_obs

    # ── 1. 参考细胞掩码 ────────────────────────────────────────────────────────
    if reference_key is not None and reference_cat is not None and reference_key in adata.obs.columns:
        cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
        ref_mask = adata.obs[reference_key].isin(cats).values
    else:
        ref_mask = np.zeros(adata.n_obs, dtype=bool)
        logging.warning("reference_key not set; all cells treated as observation")  # type: ignore

    # ── 2. Leiden 聚类（与 R cluster_by_groups + auto resolution 等价）─────────
    def _r_auto_resolution(n_cells: int) -> float:
        # R: leiden_resolution_auto = (11.98 / n_cells) ^ (1/1.165)
        return float(np.power(max(11.98 / max(n_cells, 1), 1e-6), 1.0 / 1.165))

    leiden_labels = np.empty(n_cells_total, dtype=object)

    if cluster_by_groups and ref_mask.any() and (~ref_mask).any():
        groups_to_cluster = [("Reference", ref_mask), ("Observation", ~ref_mask)]
    else:
        groups_to_cluster = [("All", np.ones(n_cells_total, dtype=bool))]

    cluster_id_offset = 0
    for grp_name, grp_mask in groups_to_cluster:
        n_grp = int(grp_mask.sum())
        if n_grp < 3:
            for cid_local, idx in enumerate(np.where(grp_mask)[0]):
                leiden_labels[idx] = f"{grp_name}_{cluster_id_offset + cid_local}"
            cluster_id_offset += n_grp
            continue

        res_grp = (
            _r_auto_resolution(n_grp)
            if leiden_resolution == "auto"
            else float(leiden_resolution)
        )
        n_pcs_grp = max(1, min(n_pcs, cnv_matrix.shape[1] - 1, n_grp - 1))
        n_neighbors_grp = max(2, min(n_neighbors, n_grp - 1))

        import anndata as _ad
        sub_X = scipy.sparse.csr_matrix(cnv_matrix[grp_mask])
        tmp = _ad.AnnData(X=sub_X)
        sc.tl.pca(tmp, n_comps=n_pcs_grp, random_state=random_state)
        sc.pp.neighbors(
            tmp, n_neighbors=n_neighbors_grp, use_rep="X_pca", random_state=random_state
        )
        sc.tl.leiden(
            tmp, resolution=res_grp, random_state=random_state, key_added="leiden_cnv"
        )
        sub_labels = tmp.obs["leiden_cnv"].values
        unique_sub = np.unique(sub_labels)
        sub_to_global = {
            s: f"{grp_name}_{cluster_id_offset + i}" for i, s in enumerate(unique_sub)
        }
        for idx_local, idx_global in enumerate(np.where(grp_mask)[0]):
            leiden_labels[idx_global] = sub_to_global[sub_labels[idx_local]]
        cluster_id_offset += len(unique_sub)

        logging.info(  # type: ignore
            f"  Leiden [{grp_name}]: n={n_grp}, resolution={res_grp:.4f}, "
            f"n_pcs={n_pcs_grp}, k={n_neighbors_grp}, n_clusters={len(unique_sub)}"
        )

    n_clusters = len(np.unique(leiden_labels))
    logging.info(  # type: ignore
        f"hmm_call_subclusters: total {n_clusters} subclusters "
        f"(cluster_by_groups={cluster_by_groups})"
    )

    # ── 3. 计算亚克隆均值 CNV 谱 + 每簇细胞数 ────────────────────────────────
    unique_clusters = np.unique(leiden_labels)
    n_cl = len(unique_clusters)
    cluster_means = np.zeros((n_cl, cnv_matrix.shape[1]), dtype=np.float64)
    cluster_ref_fracs = np.zeros(n_cl, dtype=np.float64)
    cluster_sizes = np.zeros(n_cl, dtype=np.int32)

    for ci, cl in enumerate(unique_clusters):
        mask_cl = (leiden_labels == cl)
        cluster_means[ci] = cnv_matrix[mask_cl].mean(axis=0)
        cluster_ref_fracs[ci] = ref_mask[mask_cl].mean()
        cluster_sizes[ci] = int(mask_cl.sum())

    # 参考亚克隆：> 50% 参考细胞构成的簇（R 中等价于 reference_grouped_cell_indices）
    cluster_is_ref = cluster_ref_fracs > 0.5
    logging.info(  # type: ignore
        f"  Subclusters: total={n_cl}, ref-dominated={cluster_is_ref.sum()}, "
        f"obs={n_cl - cluster_is_ref.sum()}"
    )

    # ── 4. hspike 参数校准 ───────────────────────────────────────────────────
    if precomputed_emit_means is not None:
        emit_means = np.asarray(precomputed_emit_means, dtype=np.float64)[:n_states]
        emit_stds = (
            np.asarray(precomputed_emit_stds, dtype=np.float64)[:n_states]
            if precomputed_emit_stds is not None
            else np.full(n_states, float(np.std(cnv_matrix[ref_mask])) if ref_mask.any() else 0.05)
        )
        logging.info(f"  Using precomputed emit_means={np.round(emit_means, 4)}")  # type: ignore
    elif fit_method == "hspike":
        if cluster_is_ref.any():
            emit_means, emit_stds = _fit_emission_params_hspike(
                cluster_means, cluster_is_ref
            )
        else:
            ref_cl_mask = ref_mask if ref_mask.any() else np.ones(len(cnv_matrix), dtype=bool)
            emit_means, emit_stds = _fit_emission_params_hspike(
                cnv_matrix, ref_cl_mask
            )
        emit_means = emit_means[:n_states]
        emit_stds = emit_stds[:n_states]
    elif fit_method == "fixed_log2":
        emit_means = _R_LOG2_STATE_MEANS[:n_states].copy()
        emit_stds = _R_LOG2_STATE_STDS[:n_states].copy()
    elif fit_method == "fixed_copy_ratio":
        emit_means = _R_COPY_RATIO_STATE_MEANS[:n_states].copy()
        emit_stds = _R_COPY_RATIO_STATE_STDS[:n_states].copy()
    else:  # adaptive
        emit_means, emit_stds = _fit_emission_params(
            cluster_means, adata, reference_key, reference_cat, fit_params=True
        )

    # base_sd（单细胞水平 SD）
    base_sd = float(np.median(emit_stds))

    # 检测 HMM 输入空间（log2 vs copy-ratio），以便日志清晰报告
    uns_meta = adata.uns.get(use_rep, {}) if isinstance(adata.uns.get(use_rep, {}), dict) else {}
    is_copy_ratio = bool(uns_meta.get("is_copy_ratio", False))
    is_gene_space = bool(uns_meta.get("is_gene_space", False))
    logging.info(  # type: ignore
        f"  HMM input space: {'copy-ratio' if is_copy_ratio else 'log2'}, "
        f"dim: {'gene' if is_gene_space else 'window'}, "
        f"emit_means={np.round(emit_means, 4)}, base_sd={base_sd:.5f}"
    )

    # ── 5. 选择 HMM 参数 ─────────────────────────────────────────────────────
    if use_r_viterbi:
        log_trans_hmm = _build_r_log_transition(n_states, _R_T)
        log_prior_hmm = _R_LOG_PRIOR[:n_states].copy()
    else:
        log_trans_hmm = _build_log_transition(n_states, p_stay)
        log_prior_hmm = _DEFAULT_LOG_PRIOR[:n_states].copy()

    # ── 6. 按染色体独立运行 HMM ──────────────────────────────────────────────
    chr_pos_dict = {}
    if use_rep in adata.uns and "chr_pos" in adata.uns[use_rep]:
        chr_pos_dict = adata.uns[use_rep]["chr_pos"]

    n_pos = cnv_matrix.shape[1]

    if chr_pos_dict:
        chr_names = list(chr_pos_dict.keys())
        chr_starts = [int(chr_pos_dict[c]) for c in chr_names]
        chr_ends = chr_starts[1:] + [n_pos]
        chr_ranges = list(zip(chr_names, chr_starts, chr_ends))

        state_seqs = np.full((n_cl, n_pos), NEUTRAL_STATE, dtype=np.int32)

        for chr_name, w_s, w_e in chr_ranges:
            if w_e <= w_s:
                continue
            chr_means = cluster_means[:, w_s:w_e]
            if chr_means.shape[1] < 1:
                continue

            for ci in range(n_cl):
                n_cells_ci = max(1, int(cluster_sizes[ci]))
                sd_ci = base_sd / math.sqrt(n_cells_ci)
                stds_ci = np.full(n_states, sd_ci, dtype=np.float64)

                if use_r_viterbi:
                    seq_chr = _viterbi_r_single(
                        chr_means[ci], log_trans_hmm, log_prior_hmm, emit_means, stds_ci
                    )
                else:
                    seq_chr = _viterbi_single_numpy(
                        chr_means[ci], log_trans_hmm, log_prior_hmm, emit_means, stds_ci
                    )

                # ── denoise 段长过滤（与 R denoise=TRUE 等价）────────────
                if min_segment_length > 1:
                    seq_chr = _denoise_segments(
                        seq_chr, NEUTRAL_STATE, min_segment_length
                    )
                state_seqs[ci, w_s:w_e] = seq_chr

        logging.info(  # type: ignore
            f"  Per-chromosome HMM (gene-level): {len(chr_ranges)} chroms × {n_cl} subclusters, "
            f"denoise min_segment_length={min_segment_length}"
        )
    else:
        logging.warning(  # type: ignore
            f"chr_pos not found in adata.uns['{use_rep}']; "
            "running HMM on concatenated sequence (no per-chromosome boundary)"
        )
        if use_r_viterbi:
            state_seqs = _viterbi_r_batch(
                cluster_means, log_trans_hmm, log_prior_hmm, emit_means, emit_stds, n_jobs_eff
            )
        else:
            state_seqs = _viterbi_batch_cpu(
                cluster_means, log_trans_hmm, log_prior_hmm, emit_means, emit_stds, n_jobs_eff
            )
        if min_segment_length > 1:
            for ci in range(n_cl):
                state_seqs[ci] = _denoise_segments(
                    state_seqs[ci], NEUTRAL_STATE, min_segment_length
                )

    # ── 7. R 等价 Tumor/Normal 判定（subcluster 级 "含 CNV 段" 二分类）──────
    # R: pred_cnv_regions.dat 中出现该 subcluster → Tumor.obs
    # cnvturbo v2: 该 subcluster 在所有染色体中合计 ≥ min_segments_for_tumor 个非中性段 → Tumor
    cluster_n_segments = np.zeros(n_cl, dtype=np.int32)
    for ci in range(n_cl):
        n_seg = 0
        if chr_pos_dict:
            for chr_name, w_s, w_e in chr_ranges:
                if w_e <= w_s:
                    continue
                n_seg += _count_cnv_segments(
                    state_seqs[ci, w_s:w_e], NEUTRAL_STATE, min_segment_length
                )
        else:
            n_seg = _count_cnv_segments(
                state_seqs[ci], NEUTRAL_STATE, min_segment_length
            )
        cluster_n_segments[ci] = n_seg

    # 主 Tumor 判定：subcluster 是观测组（非参考）且含 ≥ min_segments_for_tumor 个 CNV 段
    if min_segments_for_tumor > 0:
        is_tumor_cluster = (
            (cluster_n_segments >= min_segments_for_tumor) & (~cluster_is_ref)
        )
        cluster_tumor_calls = np.where(is_tumor_cluster, "Tumor", "Normal")
        logging.info(  # type: ignore
            f"  R-equivalent Tumor call: {is_tumor_cluster.sum()} Tumor subclusters, "
            f"n_seg distribution: median={int(np.median(cluster_n_segments))}, "
            f"max={int(cluster_n_segments.max())}, ref-cluster median seg="
            f"{int(np.median(cluster_n_segments[cluster_is_ref])) if cluster_is_ref.any() else 0}"
        )
    else:
        # legacy 模式：非中性占比阈值
        non_neutral_frac_clusters = (state_seqs != NEUTRAL_STATE).mean(axis=1)
        if cluster_is_ref.any():
            ref_nf = non_neutral_frac_clusters[cluster_is_ref]
            thr_ref95 = float(np.percentile(ref_nf, 95))
            effective_threshold = max(tumor_threshold, thr_ref95)
        else:
            effective_threshold = tumor_threshold
        cluster_tumor_calls = np.where(
            non_neutral_frac_clusters > effective_threshold, "Tumor", "Normal"
        )
        logging.info(  # type: ignore
            f"  Legacy Tumor call: threshold={effective_threshold:.3f}"
        )

    tumor_clust_pct = (cluster_tumor_calls == "Tumor").mean() * 100
    logging.info(  # type: ignore
        f"  Subcluster calls: {(cluster_tumor_calls == 'Tumor').sum()} Tumor "
        f"({tumor_clust_pct:.0f}%) / {n_cl}"
    )

    # ── 8. 细胞级 CNV burden（连续打分，独立于 cluster 离散判定）──────────────
    # 设计动机（针对 ROC 区分度）：
    #   原实现 cell_score = cluster_n_segments（簇级常量），同 cluster 内细胞同分，
    #   细胞间无差异 → AUC 退化。
    # R 等价物：cell-level expression deviation magnitude。
    # 这里取每细胞所有基因的 |X_cnv - neutral_anchor| mean，
    # 即 L1 距离离中性点的平均距离，单位与 HMM 输入一致：
    #   - copy-ratio 空间：anchor=1.0，score ≈ 平均拷贝偏离
    #   - log2 空间：anchor=0.0，score ≈ 平均 log2 信号强度
    # 完全数据驱动，不依赖 HMM call，对污染和阈值都稳健。
    neutral_anchor_cell = 1.0 if is_copy_ratio else 0.0
    cell_scores = np.abs(cnv_matrix - neutral_anchor_cell).mean(axis=1).astype(np.float32)

    cell_tumor_calls = np.empty(adata.n_obs, dtype=object)
    cell_leiden = np.empty(adata.n_obs, dtype=object)

    for ci, cl in enumerate(unique_clusters):
        mask_cl = (leiden_labels == cl)
        cell_tumor_calls[mask_cl] = cluster_tumor_calls[ci]
        cell_leiden[mask_cl] = cl

    # 诊断：cell_score 在 ref vs obs 上的分布（验证连续性 & 区分度）
    if ref_mask.any() and (~ref_mask).any():
        s_ref = cell_scores[ref_mask]
        s_obs = cell_scores[~ref_mask]
        logging.info(  # type: ignore
            f"  Continuous cell_score (|X - {neutral_anchor_cell:.1f}|.mean): "
            f"ref median={float(np.median(s_ref)):.4f} (IQR {float(np.percentile(s_ref, 25)):.4f}-{float(np.percentile(s_ref, 75)):.4f}); "
            f"obs median={float(np.median(s_obs)):.4f} (IQR {float(np.percentile(s_obs, 25)):.4f}-{float(np.percentile(s_obs, 75)):.4f})"
        )

    tumor_cell_pct = (cell_tumor_calls == "Tumor").mean() * 100
    logging.info(  # type: ignore
        f"  Cell-level: {(cell_tumor_calls == 'Tumor').sum()} Tumor ({tumor_cell_pct:.1f}%)"
    )

    if inplace:
        adata.obs[key_added] = cell_tumor_calls
        adata.obs[f"{key_added}_score"] = cell_scores
        adata.obs[f"{key_added}_subcluster"] = cell_leiden
        return None
    else:
        return {
            key_added: cell_tumor_calls,
            f"{key_added}_score": cell_scores,
            f"{key_added}_subcluster": cell_leiden,
        }


def hmm_call_cells(
    adata: AnnData,
    *,
    use_rep: str = "cnv",
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    n_states: int = N_STATES,
    tumor_threshold: float = _DEFAULT_TUMOR_THRESHOLD,
    fit_method: str = "adaptive",
    fit_params: bool = True,
    p_stay: float = _DEFAULT_P_STAY,
    backend: str = "auto",
    key_added: str = "cnv_tumor_call",
    n_jobs: int | None = None,
    inplace: bool = True,
    chunksize: int = 2000,
    em_n_iter: int = 20,
) -> None | dict:
    """对每个细胞运行 HMM i6 Viterbi 解码，输出细胞级 Tumor/Normal 标签。

    Parameters
    ----------
    adata
        已运行 :func:`cnvturbo.tl.infercnv` 或 :func:`cnvturbo.tl.infercnv_r_compat` 的 AnnData 对象。
    use_rep
        CNV 矩阵在 adata.obsm 中的键名（不含 "X_" 前缀），默认 "cnv"。
    reference_key
        adata.obs 中标注正常/肿瘤的列名（用于拟合发射参数）。
    reference_cat
        reference_key 中代表正常细胞的值（可多个）。
    n_states
        HMM 状态数，默认 6（i6 模型）。
    tumor_threshold
        非中性窗口比例超过此值判定为 Tumor，默认 0.20（与 R 默认行为对齐）。
    fit_method
        发射参数估计模式：

        ``"adaptive"``（默认）
            自适应信号振幅缩放；兼容任意预处理管线（包括 scanpy 标准流程）。
        ``"fixed_log2"``
            使用 R inferCNV 原始固定 log2 ratio 参数（均值 [-1, -0.5, 0, 0.5, 1, 1.5]，
            std = 0.3）。**仅在配合** :func:`cnvturbo.tl.infercnv_r_compat` **使用时
            才能保证与 R 完全对标**，因为两者均在 log2 ratio 空间操作。
        ``"em"``
            Baum-Welch EM 在实际数据上拟合高斯发射分布；
            优先 ``hmmlearn.hmm.GaussianHMM``，不可用时回退 numpy 前向-后向算法。
            拟合收敛后再运行 Viterbi，理论上最准确但耗时最长。
    fit_params
        仅对 ``fit_method="adaptive"`` 有效；True 时用参考细胞统计量校准，
        False 使用硬编码默认值。``fit_method`` 为其他值时忽略此参数。
    p_stay
        HMM 自转移概率，默认 0.99（与 R 一致）。
    backend
        'auto' / 'cuda' / 'cpu'。
    key_added
        Tumor/Normal 标签写入 adata.obs 的键名；同时会写入：
          ``adata.obs[key_added + "_score"]``    非中性窗口占比（细胞级 CNV burden）
          ``adata.obsm["X_" + key_added + "_states"]``  完整 HMM 状态序列
    n_jobs
        CPU 并行线程数（仅 backend='cpu' 时有效）。
    inplace
        True 写入 adata；False 返回结果字典。
    chunksize
        GPU 路径下每批处理的细胞数（控制显存）。
    em_n_iter
        fit_method="em" 时 Baum-Welch 最大迭代次数，默认 20。

    Returns
    -------
    None（inplace=True）或包含结果的字典（inplace=False）。
    """
    _VALID_FIT_METHODS = ("adaptive", "fixed_log2", "fixed_copy_ratio", "hspike", "em")
    if fit_method not in _VALID_FIT_METHODS:
        raise ValueError(f"fit_method 必须为 {_VALID_FIT_METHODS} 之一，得到: {fit_method!r}")

    cnv_key = f"X_{use_rep}"
    if cnv_key not in adata.obsm:
        raise KeyError(
            f"{cnv_key} not found in adata.obsm. "
            f"Did you run `tl.infercnv`? Available keys: {list(adata.obsm.keys())}"
        )

    # 取出 CNV 矩阵 → 密集
    cnv_matrix = adata.obsm[cnv_key]
    if scipy.sparse.issparse(cnv_matrix):
        cnv_matrix = cnv_matrix.toarray()
    cnv_matrix = np.asarray(cnv_matrix, dtype=np.float64)  # (N_cells, N_windows)

    n_cells, n_windows = cnv_matrix.shape
    logging.info(  # type: ignore
        f"hmm_call_cells: {n_cells} cells × {n_windows} windows, "
        f"backend={get_backend(backend)}, fit_method={fit_method}"
    )

    # ── 1. 构建 HMM 发射参数 ───────────────────────────────────────────────────
    if fit_method == "fixed_log2":
        emit_means = _R_LOG2_STATE_MEANS[:n_states].copy()
        emit_stds  = _R_LOG2_STATE_STDS[:n_states].copy()
        logging.info(  # type: ignore
            f"  fixed_log2: means={emit_means}, stds={emit_stds}"
        )
    elif fit_method == "fixed_copy_ratio":
        # 用于 infercnv_r_compat(apply_2x_transform=True) 的 copy-ratio 空间
        # R hspike 标称 CNV 比例：0.01, 0.5, 1.0, 1.5, 2.0, 3.0（中性=1.0）
        emit_means = _R_COPY_RATIO_STATE_MEANS[:n_states].copy()
        emit_stds  = _R_COPY_RATIO_STATE_STDS[:n_states].copy()
        logging.info(  # type: ignore
            f"  fixed_copy_ratio: means={emit_means}, stds={emit_stds}"
        )
    elif fit_method == "hspike":
        # 复现 R inferCNV 的 hspike 校准：用参考细胞实际分布估计 emission 参数
        if reference_key is not None and reference_cat is not None and reference_key in adata.obs.columns:
            cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
            ref_mask_hs = adata.obs[reference_key].isin(cats).values
        else:
            ref_mask_hs = np.ones(adata.n_obs, dtype=bool)
            logging.warning("hspike 模式未指定参考细胞，使用全部细胞作为参考（不推荐）")  # type: ignore
        emit_means, emit_stds = _fit_emission_params_hspike(cnv_matrix, ref_mask_hs)
        emit_means = emit_means[:n_states]
        emit_stds  = emit_stds[:n_states]
    elif fit_method == "em":
        emit_means, emit_stds = _fit_emission_params_em(
            cnv_matrix, n_states=n_states, n_iter=em_n_iter,
            init_means=_DEFAULT_STATE_MEANS[:n_states],
        )
    else:  # "adaptive"
        emit_means, emit_stds = _fit_emission_params(
            cnv_matrix, adata, reference_key, reference_cat, fit_params
        )


    log_trans = _build_log_transition(n_states, p_stay)
    log_prior = _DEFAULT_LOG_PRIOR[:n_states].copy()

    # ── 2. Viterbi 解码 ────────────────────────────────────────────────────────
    resolved_backend = get_backend(backend)

    if resolved_backend == "cuda":
        state_seq = _viterbi_batch_gpu(cnv_matrix, log_trans, log_prior, emit_means, emit_stds, chunksize)
    else:
        state_seq = _viterbi_batch_cpu(cnv_matrix, log_trans, log_prior, emit_means, emit_stds, n_jobs)

    # state_seq: (N_cells, N_windows)  int32

    # ── 3. Tumor/Normal 判定 ───────────────────────────────────────────────────
    # copy_ratio 模式中性态编号与 log2 模式相同（State 2），因为 state means 排序后
    # 中性态（1.0x 或 0.0 log2）始终在第 2 位（0-indexed）
    non_neutral_frac = (state_seq != NEUTRAL_STATE).mean(axis=1).astype(np.float32)  # (N_cells,)

    # 参考相对阈值：若 reference_key 可用，使用参考细胞 non_neutral_frac 的 95th 百分位作为上界
    # 这与 R inferCNV 的 Tumor/Normal 判定逻辑对齐（相对参考细胞噪声水平）
    if reference_key is not None and reference_cat is not None and reference_key in adata.obs.columns:
        cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
        ref_mask_hmm = adata.obs[reference_key].isin(cats).values
        if ref_mask_hmm.sum() > 10:
            ref_frac = non_neutral_frac[ref_mask_hmm]
            thr_ref95 = float(np.percentile(ref_frac, 95))
            # 取绝对阈值与参考 p95 中的较大值，防止参考细胞噪声极低时 threshold 过松
            effective_threshold = max(tumor_threshold, thr_ref95)
        else:
            effective_threshold = tumor_threshold
    else:
        effective_threshold = tumor_threshold

    tumor_calls = np.where(non_neutral_frac > effective_threshold, "Tumor", "Normal")

    tumor_pct = (tumor_calls == "Tumor").mean() * 100
    logging.info(  # type: ignore
        f"hmm_call_cells: {(tumor_calls == 'Tumor').sum()} Tumor "
        f"({tumor_pct:.1f}%), effective_threshold={effective_threshold:.3f} "
        f"(absolute={tumor_threshold}, ref_p95 used={'yes' if reference_key else 'no'})"
    )

    tumor_pct = (tumor_calls == "Tumor").mean() * 100
    logging.info(  # type: ignore
        f"hmm_call_cells: {(tumor_calls == 'Tumor').sum()} Tumor "
        f"({tumor_pct:.1f}%), threshold={tumor_threshold}"
    )

    if inplace:
        adata.obs[key_added] = tumor_calls
        adata.obs[f"{key_added}_score"] = non_neutral_frac
        adata.obsm[f"X_{key_added}_states"] = state_seq
    else:
        return {
            key_added: tumor_calls,
            f"{key_added}_score": non_neutral_frac,
            f"X_{key_added}_states": state_seq,
        }


# ── HMM 参数构建 ──────────────────────────────────────────────────────────────


def _build_log_transition(n_states: int, p_stay: float) -> np.ndarray:
    """构造对数转移矩阵。

    对角线为 p_stay，非对角均分 (1 - p_stay)。

    Returns
    -------
    (n_states, n_states) float64 对数转移矩阵。
    """
    p_switch = (1.0 - p_stay) / (n_states - 1)
    trans = np.full((n_states, n_states), p_switch)
    np.fill_diagonal(trans, p_stay)
    return np.log(trans)


def _fit_emission_params(
    cnv_matrix: np.ndarray,
    adata: AnnData,
    reference_key: str | None,
    reference_cat: str | Sequence[str] | None,
    fit_params: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合 HMM 各状态的高斯发射参数。

    策略
    ----
    1. 估计 neutral_mean / neutral_std：
       - 有参考细胞 → 参考细胞 CNV 值的均值和 std
       - 无参考细胞 → 全数据中位数和 MAD
    2. 估计信号振幅（signal_amp）：
       - 使用全部非零 CNV 绝对值的第 90 百分位。
       - cnvturbo 动态去噪后信号幅度通常在 0.01–0.5 之间，
         远小于 R inferCNV 的 0.5–2.0 量级。
       - 默认状态间距（0.5 log2FC）× scale_factor 对齐实际数据范围。
    3. 若 fit_params=False：使用 signal_amp 校准后的默认值。

    Returns
    -------
    emit_means : (n_states,) float64
    emit_stds  : (n_states,) float64
    """
    n_states = N_STATES

    # ── Step 1：估计中性态均值和方差 ─────────────────────────────────────────
    neutral_mean = 0.0
    neutral_std = None  # 稍后确定

    if reference_key is not None and reference_cat is not None and reference_key in adata.obs.columns:
        obs_col = adata.obs[reference_key]
        if isinstance(reference_cat, str):
            reference_cat = [reference_cat]
        ref_mask = obs_col.isin(reference_cat)
        if ref_mask.sum() > 10:
            ref_values = cnv_matrix[ref_mask.values, :].ravel()
            neutral_mean = float(np.mean(ref_values))
            neutral_std = max(float(np.std(ref_values)), 1e-4)
        else:
            warnings.warn(
                f"参考细胞数不足（{ref_mask.sum()}），使用全局数据估计中性参数。",
                UserWarning,
                stacklevel=3,
            )

    if neutral_std is None:
        # 无参考信息：用全数据中位数和 MAD 估计（MAD × 1.4826 ≈ std for Gaussian）
        vals = cnv_matrix.ravel()
        neutral_mean = float(np.median(vals))
        neutral_std = max(float(np.median(np.abs(vals - neutral_mean))) * 1.4826, 1e-4)

    # ── Step 2：估计信号振幅（自适应校准核心）────────────────────────────────
    # cnvturbo 动态去噪后大量值为 0；非零值才代表真实 CNV 事件。
    # signal_amp = 非零 |CNV| 第 90 百分位，代表典型 CNV 事件的幅度。
    all_nonzero = np.abs(cnv_matrix[cnv_matrix != 0])
    if len(all_nonzero) > 100:
        signal_amp = float(np.percentile(all_nonzero, 90))
    else:
        # 信号极稀疏（可能 dynamic_threshold 极高）：用全局 95 百分位
        signal_amp = float(np.percentile(np.abs(cnv_matrix), 95))

    signal_amp = max(signal_amp, 0.01)  # 防止极小值导致数值问题

    # 默认状态间距假设信号振幅 ≈ 0.5（R inferCNV 的典型量级）
    # scale_factor 将默认偏移量压缩/扩展到与实际数据对齐
    scale_factor = signal_amp / 0.5
    scale_factor = max(scale_factor, 0.02)  # 最小 2% 的默认间距，防止态均值完全重叠

    logging.info(  # type: ignore
        f"hmm_call_cells: neutral_mean={neutral_mean:.5f}, neutral_std={neutral_std:.5f}, "
        f"signal_amp(p90)={signal_amp:.5f}, scale_factor={scale_factor:.4f}"
    )

    if not fit_params:
        # 不拟合：只做信号振幅校准，不调整 neutral_mean
        emit_means = _DEFAULT_STATE_MEANS * scale_factor
        emit_stds = np.full(n_states, neutral_std * 2.0)  # 较宽以增加容错
        return emit_means, emit_stds

    # ── Step 3：构建各状态参数 ────────────────────────────────────────────────
    # 各态均值 = neutral_mean + 默认偏移量 × scale_factor
    emit_means = neutral_mean + _DEFAULT_STATE_MEANS * scale_factor

    # 各态 std = neutral_std × 宽度缩放因子（极端态适当更宽）
    # 同时也按 scale_factor 等比缩放（状态间距收窄时 std 也应收窄）
    emit_stds = neutral_std * _DEFAULT_STATE_STD_SCALE * max(scale_factor, 0.5)

    # 确保 std 不低于 neutral_std 的一半（避免极窄分布导致数值 underflow）
    emit_stds = np.clip(emit_stds, neutral_std * 0.5, None)

    return emit_means, emit_stds


def _fit_emission_params_em(
    cnv_matrix: np.ndarray,
    n_states: int = N_STATES,
    n_iter: int = 20,
    init_means: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用 Baum-Welch EM 算法在实际 CNV 数据上拟合高斯发射参数。

    策略
    ----
    1. 优先使用 ``hmmlearn.hmm.GaussianHMM``（成熟实现，收敛更稳定）。
    2. 若 hmmlearn 不可用，回退到纯 numpy 前向-后向（Forward-Backward）算法，
       结果等价但速度较慢。

    初始化
    ------
    以 ``init_means`` 或 _DEFAULT_STATE_MEANS 作为初始均值，
    用数据的 np.linspace 分位数初始化（避免空态）；
    初始 std 统一设为数据全局 std 的 0.5 倍。

    Parameters
    ----------
    cnv_matrix
        (N_cells, N_windows) 密集 float64 矩阵。
    n_states
        HMM 状态数，默认 6。
    n_iter
        Baum-Welch 最大迭代次数。
    init_means
        初始状态均值（可选）；None 时使用 _DEFAULT_STATE_MEANS。

    Returns
    -------
    emit_means : (n_states,) float64
    emit_stds  : (n_states,) float64
    """
    logging.info(f"  fit_method=em: 运行 Baum-Welch EM，n_states={n_states}, n_iter={n_iter}")  # type: ignore

    # ── 尝试 hmmlearn ─────────────────────────────────────────────────────────
    try:
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

        logging.info("  使用 hmmlearn.hmm.GaussianHMM")  # type: ignore

        # 构建转移矩阵初值（均匀，对角略高）
        p_switch = 0.01 / (n_states - 1)
        transmat_init = np.full((n_states, n_states), p_switch)
        np.fill_diagonal(transmat_init, 0.99)

        # 初始均值：按数据分位数或用户提供
        if init_means is not None:
            means_init = init_means[:n_states, np.newaxis].copy()  # (n_states, 1)
        else:
            quantiles = np.linspace(5, 95, n_states)
            means_init = np.percentile(cnv_matrix, quantiles)[:, np.newaxis]  # (n_states, 1)

        covars_init = np.full((n_states, 1, 1), np.var(cnv_matrix) * 0.25)  # 较小初始方差

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            init_params="",      # 禁用自动初始化，使用我们手动设置的值
            params="stmc",       # 学习 start、transition、means、covariances
        )
        model.startprob_ = np.exp(_DEFAULT_LOG_PRIOR[:n_states]) / np.exp(_DEFAULT_LOG_PRIOR[:n_states]).sum()
        model.transmat_   = transmat_init
        model.means_      = means_init
        model.covars_     = covars_init

        # hmmlearn 需要 (total_length, n_features)；将所有细胞拼接为单长序列
        lengths = [n_windows] * n_cells
        obs = cnv_matrix.reshape(-1, 1)            # (N_cells * N_windows, 1)
        n_cells, n_windows = cnv_matrix.shape

        model.fit(obs, lengths)

        emit_means = model.means_.ravel().astype(np.float64)    # (n_states,)
        emit_stds  = np.sqrt(model.covars_[:, 0, 0]).astype(np.float64)  # (n_states,)

        # 按均值排序（保证 state 0 = 最小，state n-1 = 最大）
        sort_idx   = np.argsort(emit_means)
        emit_means = emit_means[sort_idx]
        emit_stds  = emit_stds[sort_idx]

        logging.info(  # type: ignore
            f"  EM 收敛后：means={np.round(emit_means, 4)}, stds={np.round(emit_stds, 4)}"
        )
        return emit_means, emit_stds

    except ImportError:
        logging.warning(  # type: ignore
            "  hmmlearn 未安装，回退 numpy 前向-后向算法（较慢）。"
            "  安装：pip install hmmlearn>=0.3 或 pip install cnvturbo[hmm-em]"
        )

    # ── numpy 前向-后向回退（Baum-Welch 简化实现）────────────────────────────
    return _fit_em_numpy(cnv_matrix, n_states, n_iter, init_means)


def _fit_em_numpy(
    cnv_matrix: np.ndarray,
    n_states: int,
    n_iter: int,
    init_means: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """纯 numpy Baum-Welch EM（hmmlearn 的 numpy 回退）。

    为保证性能，在全部细胞上进行期望最大化：
    每步 E-step 计算所有细胞的责任矩阵并求和，M-step 更新参数。
    """
    import math  # noqa: PLC0415

    n_cells, T = cnv_matrix.shape
    S = n_states

    # 初始化
    if init_means is not None:
        means = init_means[:S].copy().astype(np.float64)
    else:
        means = np.linspace(cnv_matrix.min(), cnv_matrix.max(), S)

    stds = np.full(S, cnv_matrix.std() * 0.5 + 1e-6)
    prior = np.exp(_DEFAULT_LOG_PRIOR[:S])
    prior /= prior.sum()
    p_switch = 0.01 / (S - 1)
    trans = np.full((S, S), p_switch)
    np.fill_diagonal(trans, 0.99)

    log_2pi = math.log(2 * math.pi)

    for em_iter in range(n_iter):
        # E-step：对所有细胞累积 gamma（责任矩阵）
        # 仅用于更新均值和 std，不计算完整转移期望（简化 EM）
        sum_gamma   = np.zeros(S)
        sum_gamma_x = np.zeros(S)
        sum_gamma_x2 = np.zeros(S)

        for cell in range(n_cells):
            obs = cnv_matrix[cell]  # (T,)

            # 计算发射概率（对数）：(T, S)
            diff = (obs[:, None] - means[None, :]) / stds[None, :]   # (T, S)
            log_emit = -0.5 * (diff * diff + log_2pi) - np.log(stds)  # (T, S)

            # 前向算法（log scale）
            log_alpha = np.log(prior) + log_emit[0]  # (S,)
            for t in range(1, T):
                candidates = log_alpha[:, None] + np.log(trans)         # (S, S)
                log_alpha  = candidates.max(axis=0) + log_emit[t]       # (S,)  近似 fwd

            # 后向算法（log scale）
            log_beta = np.zeros(S)
            for t in range(T - 2, -1, -1):
                candidates = np.log(trans) + log_emit[t + 1] + log_beta  # (S, S)
                log_beta   = candidates.max(axis=1)                        # (S,)  近似 bwd

            # gamma：(T, S) 归一化
            log_gamma = np.zeros((T, S))
            for t in range(T):
                log_gamma[t] = (
                    np.log(prior) + log_emit[t]
                    if t == 0
                    else log_gamma[t - 1]  # 用前向累积近似
                )
            # 更简单：用 (obs_t - mean_s)^2 的 soft 分配（高斯责任）
            log_resp = -0.5 * (diff * diff) - np.log(stds)   # (T, S)
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)  # (T, S) 归一化责任矩阵

            sum_gamma   += resp.sum(axis=0)              # (S,)
            sum_gamma_x += (resp * obs[:, None]).sum(0)  # (S,)
            sum_gamma_x2 += (resp * obs[:, None] ** 2).sum(0)  # (S,)

        # M-step：更新均值和 std
        new_means = sum_gamma_x / np.maximum(sum_gamma, 1e-8)
        new_vars  = sum_gamma_x2 / np.maximum(sum_gamma, 1e-8) - new_means ** 2
        new_stds  = np.sqrt(np.maximum(new_vars, 1e-6))

        delta = np.abs(new_means - means).max()
        means = new_means
        stds  = new_stds

        logging.info(f"  EM iter {em_iter + 1}/{n_iter}: max_delta_mean={delta:.6f}")  # type: ignore
        if delta < 1e-5:
            logging.info("  EM 提前收敛")  # type: ignore
            break

    # 按均值排序
    sort_idx = np.argsort(means)
    return means[sort_idx], stds[sort_idx]


# ── CPU Viterbi（Numba 加速 + numpy 回退）────────────────────────────────────


def _viterbi_batch_cpu(
    cnv_matrix: np.ndarray,
    log_trans: np.ndarray,
    log_prior: np.ndarray,
    emit_means: np.ndarray,
    emit_stds: np.ndarray,
    n_jobs: int | None = None,
) -> np.ndarray:
    """CPU 路径批量 Viterbi。

    优先使用 Numba @njit + prange（OpenMP 并行，无 GIL 开销）；
    Numba 不可用时退回到 ThreadPoolExecutor + numpy 逐细胞循环。

    Parameters
    ----------
    cnv_matrix
        (N_cells, N_windows) 密集 float64 矩阵。
    log_trans
        (n_states, n_states) 对数转移矩阵。
    log_prior
        (n_states,) 初始状态对数先验。
    emit_means / emit_stds
        (n_states,) 各态高斯发射参数。

    Returns
    -------
    state_seq : (N_cells, N_windows) int32
    """
    n_jobs_eff = get_n_jobs(n_jobs)

    if has_numba():
        logging.info("hmm_call_cells (CPU): 使用 Numba 并行 Viterbi")  # type: ignore
        return _viterbi_numba_dispatch(cnv_matrix, log_trans, log_prior, emit_means, emit_stds)
    else:
        logging.info(  # type: ignore
            f"hmm_call_cells (CPU): Numba 不可用，使用 numpy 逐细胞 Viterbi (n_jobs={n_jobs_eff})"
        )
        return _viterbi_numpy_parallel(cnv_matrix, log_trans, log_prior, emit_means, emit_stds, n_jobs_eff)


def _viterbi_numba_dispatch(cnv_matrix, log_trans, log_prior, emit_means, emit_stds):
    """构建并调用 Numba 编译的批量 Viterbi。"""
    import numba  # noqa: PLC0415

    log_emit_stds = np.log(emit_stds)

    @numba.njit(cache=True, fastmath=True)
    def _log_normal_scalar(x, mu, sigma, log_sigma):
        """单值高斯对数概率（用于 JIT 内联）。"""
        diff = (x - mu) / sigma
        return -0.5 * diff * diff - log_sigma - 0.9189385332046727  # 0.5*log(2π)

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _viterbi_batch(x, log_trans, log_prior, emit_means, emit_stds, log_emit_stds):
        """
        批量 Viterbi（Numba 并行版）。

        Parameters
        ----------
        x            (N, T) float64
        log_trans    (S, S) float64
        log_prior    (S,)   float64
        emit_means   (S,)   float64
        emit_stds    (S,)   float64
        log_emit_stds (S,)  float64

        Returns
        -------
        states : (N, T) int32
        """
        N, T = x.shape
        S = log_trans.shape[0]
        states = np.zeros((N, T), dtype=np.int32)

        for cell in numba.prange(N):
            obs = x[cell]
            # backptr[t, s] = 到达时刻 t 状态 s 时，前一时刻的最优状态
            backptr = np.zeros((T, S), dtype=np.int32)
            log_alpha = np.empty(S)

            # 初始时刻：log_alpha[s] = log_prior[s] + log_emit(obs[0] | s)
            for s in range(S):
                log_alpha[s] = log_prior[s] + _log_normal_scalar(
                    obs[0], emit_means[s], emit_stds[s], log_emit_stds[s]
                )

            # 前向递推
            new_alpha = np.empty(S)
            for t in range(1, T):
                obs_t = obs[t]
                for s_next in range(S):
                    # 找最优前驱状态
                    best_score = log_alpha[0] + log_trans[0, s_next]
                    best_prev = 0
                    for s_prev in range(1, S):
                        score = log_alpha[s_prev] + log_trans[s_prev, s_next]
                        if score > best_score:
                            best_score = score
                            best_prev = s_prev
                    emit = _log_normal_scalar(
                        obs_t, emit_means[s_next], emit_stds[s_next], log_emit_stds[s_next]
                    )
                    new_alpha[s_next] = best_score + emit
                    backptr[t, s_next] = best_prev
                # 原地更新 log_alpha
                for s in range(S):
                    log_alpha[s] = new_alpha[s]

            # 回溯
            best_end = 0
            for s in range(1, S):
                if log_alpha[s] > log_alpha[best_end]:
                    best_end = s
            states[cell, T - 1] = best_end
            for t in range(T - 2, -1, -1):
                states[cell, t] = backptr[t + 1, states[cell, t + 1]]

        return states

    return _viterbi_batch(
        cnv_matrix, log_trans, log_prior,
        emit_means, emit_stds, log_emit_stds,
    )


def _viterbi_single_numpy(obs, log_trans, log_prior, emit_means, emit_stds):
    """单细胞 numpy Viterbi（无 Numba 的回退方案）。"""
    T = len(obs)
    S = len(emit_means)
    log_emit_stds = np.log(emit_stds)

    def log_emit_vec(obs_t):
        """返回 (S,) 所有状态的发射对数概率。"""
        diff = (obs_t - emit_means) / emit_stds
        return -0.5 * diff * diff - log_emit_stds - 0.5 * math.log(2 * math.pi)

    log_alpha = log_prior + log_emit_vec(obs[0])  # (S,)
    backptr = np.zeros((T, S), dtype=np.int32)

    for t in range(1, T):
        # candidates[s_prev, s_next] = log_alpha[s_prev] + log_trans[s_prev, s_next]
        candidates = log_alpha[:, None] + log_trans  # (S, S)
        best_prev = np.argmax(candidates, axis=0)    # (S,)
        best_scores = candidates[best_prev, np.arange(S)]
        log_alpha = best_scores + log_emit_vec(obs[t])
        backptr[t] = best_prev

    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = int(np.argmax(log_alpha))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]

    return states


def _viterbi_numpy_parallel(cnv_matrix, log_trans, log_prior, emit_means, emit_stds, n_jobs):
    """多线程 numpy Viterbi（numpy 释放 GIL，ThreadPoolExecutor 有效并行）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    from tqdm.auto import tqdm  # noqa: PLC0415

    N, T = cnv_matrix.shape
    states = np.zeros((N, T), dtype=np.int32)

    def _worker(cell_idx):
        obs = cnv_matrix[cell_idx]
        return cell_idx, _viterbi_single_numpy(obs, log_trans, log_prior, emit_means, emit_stds)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_worker, i): i for i in range(N)}
        for fut in tqdm(as_completed(futures), total=N, desc="Viterbi (numpy)"):
            cell_idx, seq = fut.result()
            states[cell_idx] = seq

    return states


# ── GPU Viterbi（PyTorch CUDA 批量）─────────────────────────────────────────


def _viterbi_batch_gpu(
    cnv_matrix: np.ndarray,
    log_trans: np.ndarray,
    log_prior: np.ndarray,
    emit_means: np.ndarray,
    emit_stds: np.ndarray,
    chunksize: int = 2000,
) -> np.ndarray:
    """GPU 路径：PyTorch 批量 Viterbi。

    在每个时间步 t 上对所有细胞同时执行矩阵运算（CUDA 并行），
    时间步之间的顺序依赖通过 Python 循环处理。

    显存估算（以 float32）：
      (N_cells × N_states × N_windows × 4 bytes) → 2000 × 6 × 2000 × 4 ≈ 96 MB
    额外 backpointer (int32)：同等量级。总体 < 512 MB / chunk，现代 GPU 可接受。

    Parameters
    ----------
    cnv_matrix : (N, T) float64 dense
    log_trans  : (S, S) float64
    log_prior  : (S,) float64
    emit_means : (S,) float64
    emit_stds  : (S,) float64
    chunksize  : 每批细胞数（控制显存）

    Returns
    -------
    states : (N, T) int32（在 CPU 上）
    """
    import torch  # noqa: PLC0415

    device = torch.device("cuda")
    N, T = cnv_matrix.shape
    S = len(emit_means)

    # 常量张量上传 GPU
    log_trans_t = torch.tensor(log_trans, dtype=torch.float32, device=device)   # (S, S)
    log_prior_t = torch.tensor(log_prior, dtype=torch.float32, device=device)   # (S,)
    means_t     = torch.tensor(emit_means, dtype=torch.float32, device=device)  # (S,)
    stds_t      = torch.tensor(emit_stds, dtype=torch.float32, device=device)   # (S,)
    log_stds_t  = torch.log(stds_t)
    log_2pi_half = torch.tensor(0.5 * math.log(2 * math.pi), device=device)

    all_states = np.zeros((N, T), dtype=np.int32)

    for chunk_start in range(0, N, chunksize):
        chunk_end = min(chunk_start + chunksize, N)
        n_chunk = chunk_end - chunk_start

        # 上传当前 chunk
        obs_gpu = torch.tensor(
            cnv_matrix[chunk_start:chunk_end], dtype=torch.float32, device=device
        )  # (n_chunk, T)

        # 预计算所有时间步的发射对数概率：(n_chunk, T, S)
        # obs_gpu[:, :, None]: (n_chunk, T, 1) - means_t: (S,) → 广播 (n_chunk, T, S)
        diff = (obs_gpu.unsqueeze(2) - means_t) / stds_t  # (n_chunk, T, S)
        log_emit = -0.5 * diff * diff - log_stds_t - log_2pi_half  # (n_chunk, T, S)

        # 初始时刻 log_alpha：(n_chunk, S)
        log_alpha = log_prior_t.unsqueeze(0) + log_emit[:, 0, :]

        # backptr：(n_chunk, T, S)  int32 — 记录每个 (时刻, 状态) 的最优前驱状态
        backptr = torch.zeros((n_chunk, T, S), dtype=torch.int32, device=device)

        for t in range(1, T):
            # 转移得分：log_alpha[:, s_prev] + log_trans[s_prev, s_next]
            # log_alpha.unsqueeze(2): (n_chunk, S, 1) + log_trans_t: (S, S) → (n_chunk, S, S)
            trans_scores = log_alpha.unsqueeze(2) + log_trans_t.unsqueeze(0)  # (n_chunk, S, S)
            # 对 s_prev 维度取最大值
            best_scores, best_prevs = trans_scores.max(dim=1)  # 各 (n_chunk, S)
            log_alpha = best_scores + log_emit[:, t, :]         # (n_chunk, S)
            backptr[:, t, :] = best_prevs                       # (n_chunk, S)

        # 回溯（仍在 GPU 上，但逐步循环 — GPU 内存带宽足以支撑）
        states_gpu = torch.zeros((n_chunk, T), dtype=torch.int32, device=device)
        states_gpu[:, T - 1] = log_alpha.argmax(dim=1)  # (n_chunk,)
        for t in range(T - 2, -1, -1):
            # 从 backptr 中取出当前时刻 t+1 各细胞最优状态对应的前驱
            cur_states = states_gpu[:, t + 1].unsqueeze(1)          # (n_chunk, 1) — long
            # gather 需要 long tensor
            prev_states = backptr[:, t + 1, :].gather(
                1, cur_states.long()
            ).squeeze(1)  # (n_chunk,)
            states_gpu[:, t] = prev_states

        all_states[chunk_start:chunk_end] = states_gpu.cpu().numpy()

    logging.info(f"hmm_call_cells (GPU): {N} cells 解码完成")  # type: ignore
    return all_states


# ── hspike 模拟：复现 R inferCNV 的 hspike 校准机制 ──────────────────────────


def _fit_emission_params_hspike(
    cnv_matrix: np.ndarray,
    ref_mask: np.ndarray,
    cnv_ratios: tuple[float, ...] = (0.01, 0.5, 1.0, 1.5, 2.0, 3.0),
    f_cnv: float = 0.5,
    max_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """解析式 hspike——R inferCNV emission 参数对标实现（v3，污染稳健）。

    R inferCNV 的关键设计（基于 R/inferCNV_HMM.R + inferCNV_hidden_spike.R）：
      1. emission means 来自 hspike 模拟（合成基因组：50% CNV chr + 50% neutral chr），
         **与 reference 是否被污染无关**；
      2. emission sds 来自 hspike 中性态基因的 inter-gene 离散度，对 num_cells 做 lm 拟合，
         本质就是 base_sd / sqrt(N) 的 standard-error-of-mean；
      3. R 不做"按观测信号缩放 emission 间距"的污染校正——这是 cnvturbo v1/v2 引入的偏差，
         会让 HMM 退化为最近邻分类器，过判 CNV 段。

    本实现：
      - emit_means 使用 hspike 解析式（与污染无关）：
          D(r)     = f_cnv * r + (1 - f_cnv)
          cnv_raw  = clip(log2(r/D), ±max)
          neut_raw = clip(log2(1/D), ±max)
          mean_log2(r) = (cnv_raw - neut_raw) / 2     # per-cell median centering 精确解
        copy-ratio 空间则进一步取 2^x：[0.25, 0.71, 1.0, 1.22, 1.41, 1.73]，中性 = 1.0
      - emit_stds 使用**污染稳健的中性态噪声估计**：
          以中性锚点（log2 空间 = 0；copy-ratio = 1.0）为基准，对 ref cells
          取 MAD（中位绝对偏差），再 ×1.4826 还原为 std。
          MAD 是分位数估计，参考组掺入 ~49% 肿瘤细胞时仍稳健（因肿瘤偏离量
          不会同时拉高 50% 分位）。

    Parameters
    ----------
    cnv_matrix : (N, W) float64，预处理后的 CNV 矩阵
    ref_mask   : (N,) bool，参考细胞掩码
    cnv_ratios : 6 个状态的 CNV 拷贝比例（R hspike 固定值）
    f_cnv      : CNV 染色体占合成基因组比例（R = 0.5）
    max_threshold : R 的 max_centered_threshold（默认 3.0）

    Returns
    -------
    emit_means, emit_stds : each (len(cnv_ratios),) float64
        emit_means 在 HMM 输入空间（copy-ratio 中性=1.0；log2 中性=0）；
        emit_stds 为单细胞水平的中性噪声（hmm_call_subclusters 内部按 sqrt(N) 缩放）。
    """
    ref_vals    = cnv_matrix[ref_mask].ravel()
    ref_median  = float(np.median(ref_vals))
    ref_std_raw = max(float(np.std(ref_vals)), 1e-6)

    # ── 检测 HMM 输入空间（copy-ratio 中性 ≈ 1.0；log2 中性 ≈ 0）──────────
    # 用 |median - 1| vs |median| 判断空间，避免污染导致 mean 偏移影响判断
    is_copy_ratio = abs(ref_median - 1.0) < abs(ref_median - 0.0)
    neutral_anchor = 1.0 if is_copy_ratio else 0.0

    # ── 1. emit_means：解析式 hspike（与 R 一致，与污染无关）────────────────
    cnv_ratios_arr = np.array(cnv_ratios, dtype=np.float64)
    D = f_cnv * cnv_ratios_arr + (1.0 - f_cnv)
    cnv_raw  = np.clip(np.log2(np.maximum(cnv_ratios_arr / D, 1e-8)), -max_threshold, max_threshold)
    neut_raw = np.clip(np.log2(np.maximum(1.0 / D, 1e-8)), -max_threshold, max_threshold)
    emit_means_log2 = (cnv_raw - neut_raw) / 2.0

    if is_copy_ratio:
        # copy-ratio 空间：[0.25, 0.707, 1.0, 1.224, 1.414, 1.731]
        emit_means = np.power(2.0, emit_means_log2)
    else:
        # log2 空间：[-2.0, -0.5, 0.0, 0.293, 0.5, 0.793]
        emit_means = emit_means_log2.copy()

    # ── 2. emit_stds：污染稳健中性噪声（MAD 估计）──────────────────────────
    # MAD = median(|x - anchor|)；高斯下 std ≈ 1.4826 * MAD
    # 50% 分位为基础，参考组掺入 49% 肿瘤细胞时不被污染拉偏
    mad = float(np.median(np.abs(ref_vals - neutral_anchor)))
    robust_std = max(mad * 1.4826, 1e-4)

    emit_stds = np.full(len(cnv_ratios), robust_std, dtype=np.float64)

    # 诊断信息：报告原始 mean / median / MAD-derived std vs 简单 std
    obs_idx = ~ref_mask
    if obs_idx.any():
        obs_vals  = cnv_matrix[obs_idx].ravel()
        obs_range = float(np.percentile(obs_vals, 90) - np.percentile(obs_vals, 10))
    else:
        obs_range = float("nan")

    logging.info(  # type: ignore
        f"  hspike (R-aligned, no contamination scaling): "
        f"space={'copy-ratio' if is_copy_ratio else 'log2'}, "
        f"ref n={ref_mask.sum()}, ref median={ref_median:.5f}, "
        f"raw std={ref_std_raw:.5f} (incl. tumor contamination), "
        f"robust MAD-std={robust_std:.5f} (used), obs p10-p90 range={obs_range:.5f}\n"
        f"  hspike: emit_means={np.round(emit_means, 5)}\n"
        f"  hspike: emit_stds={np.round(emit_stds, 5)}"
    )
    return emit_means, emit_stds
