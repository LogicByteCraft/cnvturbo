from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scanpy.datasets

from cnvturbo.io import genomic_position_from_biomart, genomic_position_from_gtf
from cnvturbo.tl import infercnv


@pytest.mark.parametrize(
    "adata,gtf,kwargs,expected_genes",
    [
        # weird gtfparse issue
        # [scanpy.datasets.pbmc3k(), "chr1_ensembl.gtf", {}, 42],
        [scanpy.datasets.pbmc3k(), "chr21_gencode.gtf", {}, 56],
        [scanpy.datasets.pbmc3k(), "chr21_gencode.gtf", {"adata_gene_id": "gene_ids", "gtf_gene_id": "gene_id"}, 116],
    ],
)
def test_get_genomic_position_from_gtf(adata, gtf, kwargs, testdata, expected_genes):
    genomic_position_from_gtf(testdata / gtf, adata, **kwargs)
    # those entries that are not null should start with "chr"
    assert all(adata.var["chromosome"].dropna().str.startswith("chr"))
    # start and end are equally populatedj
    npt.assert_array_equal(adata.var["start"].isnull().values, adata.var["end"].isnull().values)
    # most genes are covered. Matching ENSG we get almost all, using gene symbols only ~17k on this dataset
    assert np.sum(~adata.var["start"].isnull()) == expected_genes
    # can run infercnv on the result
    infercnv(adata)


def test_get_genomic_position_from_gtf_categorical_chromosome(testdata):
    """Regression test for v0.2.1.

    gtfparse >= 2.0 (and the pyarrow backend) returns Categorical / ArrowExtensionArray
    columns. The ``"chr" + col`` and ``col.str.startswith`` calls inside
    :func:`genomic_position_from_gtf` must remain robust to that. This test simulates
    the worst case (chromosome / gene_id / gene_name all Categorical without ``chr``
    prefix) by patching ``gtfparse.read_gtf`` and verifies the function still
    annotates the AnnData without raising and produces ``chr``-prefixed chromosomes.
    """
    import gtfparse

    pbmc = scanpy.datasets.pbmc3k()
    # Build a minimal GTF-like frame whose string columns are *Categorical*
    # (no `chr` prefix, so the prefix-add branch is exercised) and whose dtype
    # would have triggered "TypeError: can only concatenate str (not 'Categorical')
    # to str" in v0.2.0.
    gene_names = pbmc.var_names[:50].tolist()
    fake = pd.DataFrame(
        {
            "seqname": pd.Categorical(["1"] * len(gene_names)),
            "feature": ["gene"] * len(gene_names),
            "start": np.arange(1, len(gene_names) + 1) * 1000,
            "end": np.arange(1, len(gene_names) + 1) * 1000 + 500,
            "gene_id": pd.Categorical([f"ENSG{i:011d}" for i in range(len(gene_names))]),
            "gene_name": pd.Categorical(gene_names),
        }
    )

    class _FakeArrow:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    with patch.object(gtfparse, "read_gtf", return_value=_FakeArrow(fake)):
        genomic_position_from_gtf(testdata / "chr21_gencode.gtf", pbmc)

    annotated = pbmc.var["chromosome"].dropna()
    assert len(annotated) == len(gene_names)
    assert all(annotated.astype(str).str.startswith("chr"))


@pytest.mark.parametrize(
    "adata,kwargs",
    [
        [scanpy.datasets.pbmc3k(), {"adata_gene_id": "gene_ids"}],
        [scanpy.datasets.pbmc3k(), {"biomart_gene_id": "hgnc_symbol"}],
    ],
)
def test_get_genomic_position_from_biomart(adata, kwargs):
    genomic_position_from_biomart(adata, use_cache=True, **kwargs)
    # those entries that are not null should start with "chr"
    assert all(adata.var["chromosome"].dropna().str.startswith("chr"))
    # start and end are equally populatedj
    npt.assert_array_equal(adata.var["start"].isnull().values, adata.var["end"].isnull().values)
    # most genes are covered. Matching ENSG we get almost all, using gene symbols only ~17k on this dataset
    assert np.sum(~adata.var["start"].isnull()) > 15000
    # can run infercnv on the result
    infercnv(adata)
