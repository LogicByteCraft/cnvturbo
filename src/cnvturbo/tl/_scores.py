"""量化和评估拷贝数变异（CNV）的各类评分函数。

新增函数
--------
cnv_score_cell
    细胞级 CNV burden score，基于 HMM 状态（优先）或 |X_cnv| 均值（回退）。
    与 R inferCNV 的细胞级输出直接可比。

原有函数
--------
cnv_score   基于 leiden cluster 的聚类级均值（向后兼容）
ithgex      基于基因表达的异质性评分
ithcna      基于 CNV profile 的异质性评分
"""

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from cnvturbo._util import _choose_mtx_rep

# ── 新增：细胞级 CNV burden score ─────────────────────────────────────────────


def cnv_score_cell(
    adata: AnnData,
    *,
    use_hmm_states: bool = True,
    use_rep: str = "cnv",
    hmm_key: str = "cnv_tumor_call",
    key_added: str = "cnv_score_cell",
    neutral_state: int = 2,
    inplace: bool = True,
) -> np.ndarray | None:
    """每个细胞的 CNV burden score（细胞级，与 R inferCNV 输出可比）。

    优先级：
    1. 若已运行 :func:`cnvturbo.tl.hmm_call_cells`：
       使用 HMM 状态序列中非中性状态（state ≠ neutral_state）的窗口占比。
       这与 R inferCNV HMM 的 CNV burden 定义完全一致。
    2. 否则（无 HMM 结果）：
       使用 ``|X_{use_rep}|`` 的逐细胞均值作为代替。
       此值等同于 :func:`cnv_score` 在 cell 粒度的版本（而非 cluster 均值）。

    Parameters
    ----------
    adata
        注释数据矩阵。
    use_hmm_states
        True 时优先使用 HMM 状态序列；False 强制使用 |X_cnv| 均值。
    use_rep
        CNV 矩阵在 adata.obsm 中的键名（不含 "X_" 前缀）。
    hmm_key
        hmm_call_cells 写入结果的键名前缀（同 key_added 参数）。
    key_added
        写入 adata.obs 的列名。
    neutral_state
        HMM 中性态编号（默认 2，即 i6 模型中的二倍体中性）。
    inplace
        True 写入 adata.obs，False 返回数组。

    Returns
    -------
    inplace=True 时返回 None；否则返回 (N_cells,) float32 数组。
    """
    hmm_states_key = f"X_{hmm_key}_states"

    if use_hmm_states and hmm_states_key in adata.obsm:
        # ── 路径 1：HMM 状态序列（最准确，与 R 行为对齐）─────────────────────
        state_seq = adata.obsm[hmm_states_key]  # (N_cells, N_windows)
        if sp.issparse(state_seq):
            state_seq = state_seq.toarray()
        scores = (state_seq != neutral_state).mean(axis=1).astype(np.float32)
    else:
        # ── 路径 2：|X_cnv| 均值（HMM 未运行时的代替方案）──────────────────
        cnv_key = f"X_{use_rep}"
        if cnv_key not in adata.obsm:
            raise KeyError(
                f"{cnv_key} not found in adata.obsm. Did you run `tl.infercnv`? "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        cnv_mtx = adata.obsm[cnv_key]
        if sp.issparse(cnv_mtx):
            scores = np.array(cnv_mtx.multiply(cnv_mtx).mean(axis=1)).ravel() ** 0.5
        else:
            scores = np.abs(cnv_mtx).mean(axis=1).astype(np.float32)

    if inplace:
        adata.obs[key_added] = scores
    else:
        return scores


# ── 原有函数（向后兼容，不做修改）───────────────────────────────────────────


def cnv_score(
    adata: AnnData,
    groupby: str = "cnv_leiden",
    *,
    use_rep: str = "cnv",
    key_added: str = "cnv_score",
    inplace: bool = True,
    obs_key=None,
) -> Mapping[Any, np.number] | None:
    """Assign each cnv cluster a CNV score.

    Clusters with a high score are likely affected by copy number abberations.
    Based on this score, cells can be divided into tumor/normal cells.

    Ths score is currently simply defined as the mean of the absolute values of the
    CNV scores in each cluster.

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        Key under which the clustering is stored in adata.obs. Usually
        the result of :func:`cnvturbo.tl.leiden`, but could also be
        other grouping information, e.g. sample or patient information.
    use_rep
        Key under which the result of :func:`cnvturbo.tl.infercnv` is stored
        in adata.
    key_added
        Key under which the score will be stored in `adata.obs`.
    inplace
        If True, store the result in adata, otherwise return it.
    obs_key
        Deprecated alias for `groupby`.

    Returns
    -------
    Depending on the value of `inplace`, either returns `None` or
    dictionary with the score per group.
    """
    if obs_key is not None:
        warnings.warn(
            "The obs_key argument has been renamed to `groupby` for consistency with "
            "other functions and will be removed in the future. ",
            category=FutureWarning,
            stacklevel=2,
        )
        groupby = obs_key

    if groupby not in adata.obs.columns and groupby == "cnv_leiden":
        raise ValueError("`cnv_leiden` not found in `adata.obs`. Did you run `tl.leiden`?")
    cluster_score = {
        cluster: np.mean(np.abs(adata.obsm[f"X_{use_rep}"][adata.obs[groupby].values == cluster, :]))
        for cluster in adata.obs[groupby].unique()
    }

    if inplace:
        score_array = np.array([cluster_score[c] for c in adata.obs[groupby]])
        adata.obs[key_added] = score_array
    else:
        return cluster_score


def ithgex(
    adata: AnnData,
    groupby: str,
    *,
    use_raw: bool | None = None,
    layer: str | None = None,
    inplace: bool = True,
    key_added: str = "ithgex",
) -> Mapping[str, float] | None:
    """Compute the ITHGEX diversity score based on gene expression cite:`Wu2021`.

    A high score indicates a high diversity of gene expression profiles of cells
    within a group.

    The score is defined as follows:

        Intratumoral heterogeneity scores based on CNAs and gene expressions
        The calculations of intratumoral heterogeneity scores were inspired by a
        previous study and modified as follows. First, to calculate ITHCNA, we used
        the relative expression value matrix generated by inferCNV and calculated the
        pairwise cell–cell distances using Pearson's correlation coefficients for each
        patient. ITHCNA was defined as interquartile range (IQR) of the distribution for
        all malignant cell pairs' Pearson's correlation coefficients. **Similarly, we also
        used gene expression profiles of cancer cells of each patient to construct the
        distribution of the intratumoral distances. ITHGEX was assigned as the IQR of the
        distribution.**

        (from :cite:`Wu2021`)


    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        calculate diversity for each group defined in this category.
    use_raw
        Use gene expression from `adata.raw`. Defaut: Use `.raw` if available,
        `.X` otherwise.
    layer
        Use gene expression from `adata.layers[layer]`
    inplace
        If True, store the result in adata, otherwise return it.
    key_added
        Key under which the score will be stored in `adata.obs`.

    Returns
    -------
    Depending of the value of `inplace` either returns a dictionary
    with one value per group or `None`.
    """
    groups = adata.obs[groupby].unique()
    ithgex = {}
    for group in groups:
        tmp_adata = adata[adata.obs[groupby] == group, :]
        X = _choose_mtx_rep(tmp_adata, use_raw, layer)
        if sp.issparse(X):
            X = X.todense()
        if X.shape[0] <= 1:
            continue
        pcorr = np.corrcoef(X, rowvar=True)
        assert pcorr.shape == (
            tmp_adata.shape[0],
            tmp_adata.shape[0],
        ), f"pcorr is a cell x cell matrix {tmp_adata.shape[0]} {pcorr.shape}"
        q75, q25 = np.percentile(pcorr, [75, 25])
        ithgex[group] = q75 - q25

    if inplace:
        ithgex_obs = np.empty(adata.shape[0])
        for group in groups:
            ithgex_obs[adata.obs[groupby] == group] = ithgex[group]
        adata.obs[key_added] = ithgex_obs
    else:
        return ithgex


def ithcna(
    adata: AnnData,
    groupby: str,
    *,
    use_rep: str = "X_cnv",
    key_added: str = "ithcna",
    inplace: bool = True,
) -> Mapping[str, float] | None:
    """Compute the ITHCNA diversity score based on copy number variation :cite:`Wu2021`.

    A high score indicates a high diversity of CNV profiles of cells
    within a group.

    The score is defined as follows:

        Intratumoral heterogeneity scores based on CNAs and gene expressions
        The calculations of intratumoral heterogeneity scores were inspired by a
        previous study and modified as follows. First, to calculate ITHCNA, we used
        the relative expression value matrix generated by inferCNV and calculated the
        pairwise cell–cell distances using Pearson's correlation coefficients for each
        patient. ITHCNA was defined as interquartile range (IQR) of the distribution for
        all malignant cell pairs' Pearson's correlation coefficients.

        (from :cite:`Wu2021`)

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        calculate diversity for each group defined in this category.
    use_rep
        Key under which the result of :func:`cnvturbo.tl.infercnv` is stored
        in adata.
    key_added
        Key under which the score will be stored in `adata.obs`.
    inplace
        If True, store the result in adata, otherwise return it.

    Returns
    -------
    Depending of the value of `inplace` either returns a dictionary
    with one value per group or `None`.
    """
    groups = adata.obs[groupby].unique()
    ithcna = {}
    for group in groups:
        tmp_adata = adata[adata.obs[groupby] == group, :]
        X = tmp_adata.obsm[use_rep]
        if sp.issparse(X):
            X = X.todense()
        if X.shape[0] <= 1:
            continue
        pcorr = np.corrcoef(X, rowvar=True)
        assert pcorr.shape == (
            tmp_adata.shape[0],
            tmp_adata.shape[0],
        ), "pcorr is a cell x cell matrix"
        q75, q25 = np.percentile(pcorr, [75, 25])
        ithcna[group] = q75 - q25

    if inplace:
        ithcna_obs = np.empty(adata.shape[0])
        for group in groups:
            ithcna_obs[adata.obs[groupby] == group] = ithcna[group]
        adata.obs[key_added] = ithcna_obs
    else:
        return ithcna
