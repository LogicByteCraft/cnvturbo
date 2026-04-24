"""Example datasets."""

import importlib.resources as pkg_resources

import scanpy as sc
from anndata import AnnData
from scanpy import settings
from scanpy.readwrite import read

from . import data


def oligodendroglioma() -> AnnData:
    """The original inferCNV example dataset.

    Derived from :cite:`Tirosh2016`.
    """
    with pkg_resources.path(data, "oligodendroglioma.h5ad") as p:
        return sc.read_h5ad(p)


def maynard2020_3k() -> AnnData:
    """Return the Maynard 2020 dataset (downsampled to 3000 cells).

    .. note::

        The remote dataset is not yet hosted under this project's release assets.
        Until then, please obtain the file ``maynard2020_3k.h5ad`` separately
        and place it under ``scanpy.settings.datasetdir``.
    """
    filename = settings.datasetdir / "maynard2020_3k.h5ad"
    if not filename.exists():
        raise FileNotFoundError(
            f"{filename} not found. The bundled remote URL has not been published yet; "
            "please obtain `maynard2020_3k.h5ad` manually and place it at the path above."
        )
    return read(filename)
