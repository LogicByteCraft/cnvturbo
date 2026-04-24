"""计算后端自动检测：CUDA GPU 或 CPU。

优先级：cuda > cpu
所有需要后端路由的模块通过 get_backend() 获取当前可用后端。
"""

import warnings
from multiprocessing import cpu_count


def get_backend(backend: str = "auto") -> str:
    """返回实际使用的后端名称：'cuda' 或 'cpu'。

    Parameters
    ----------
    backend
        'auto'  自动探测：有 CUDA 则用 GPU，否则 CPU。
        'cuda'  强制使用 GPU，不可用时回退并发出警告。
        'cpu'   强制使用 CPU。

    Returns
    -------
    str
        'cuda' 或 'cpu'。
    """
    if backend == "cpu":
        return "cpu"

    if backend == "cuda":
        resolved = _try_cuda()
        if resolved == "cpu":
            warnings.warn(
                "backend='cuda' 已请求，但 CUDA 不可用或 torch 未安装，回退到 'cpu'。",
                UserWarning,
                stacklevel=2,
            )
        return resolved

    # backend == "auto"
    return _try_cuda()


def _try_cuda() -> str:
    """尝试检测 CUDA，返回 'cuda' 或 'cpu'。"""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_n_jobs(n_jobs: int | None = None) -> int:
    """返回有效 CPU 并行核数。

    Parameters
    ----------
    n_jobs
        None 或 ≤0 时自动使用全部核心。
    """
    if n_jobs is None or n_jobs <= 0:
        return cpu_count()
    return int(n_jobs)


def has_numba() -> bool:
    """检查 numba 是否可用。"""
    try:
        import numba  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


def has_torch() -> bool:
    """检查 torch 是否可用。"""
    try:
        import torch  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


def get_torch_device(backend: str) -> "torch.device":  # noqa: F821
    """返回对应的 torch.device 对象。"""
    import torch  # noqa: PLC0415

    if backend == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")
