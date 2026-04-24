"""cnvturbo: GPU/Numba-accelerated CNV inference with HMM i6 cell-level tumor calling.

对 cnvturbo 的全面升级：
  - 新增 HMM i6 细胞级 Tumor/Normal 判定（tl.hmm_call_cells）
  - 滑窗卷积 10x 加速（Numba parallel CPU + torch GPU）
  - 细胞级 CNV burden score（tl.cnv_score_cell）
  - 与原 cnvturbo API 完全兼容
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cnvturbo")
except PackageNotFoundError:
    # 开发模式下直接从 src/ 目录使用时未安装包元数据
    __version__ = "dev"

from . import datasets, io, pl, pp, tl

__all__ = ["datasets", "io", "pl", "pp", "tl"]
