from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy.queries
from anndata import AnnData
from scanpy import logging


def genomic_position_from_biomart(
    adata: AnnData | None = None,
    *,
    adata_gene_id: str | None = None,
    biomart_gene_id="ensembl_gene_id",
    species: str = "hsapiens",
    inplace: bool = True,
    **kwargs,
):
    """
    Get genomic gene positions from ENSEMBL Biomart.

    Parameters
    ----------
    adata
        Adds the genomic positions to `adata.var`. If adata is None, returns
        a data frame with the genomic positions instead.
    adata_gene_id
        Column in `adata.var` that contains (ENSMBL) gene IDs. If not specified,
        use `adata.var_names`.
    biomart_gene_id
        The biomart column to use as gene identifier. Typically this would be `ensembl_gene_id` or `hgnc_symbol`,
        but could be different for other species.
    inplace
        If True, add the annotations directly to adata, otherwise return a dataframe.
    **kwargs
        passed on to :func:`scanpy.queries.biomart_annotations`
    """
    biomart_annot = (
        scanpy.queries.biomart_annotations(
            species,
            [
                biomart_gene_id,
                "start_position",
                "end_position",
                "chromosome_name",
            ],
            **kwargs,
        )
        .rename(
            columns={
                "start_position": "start",
                "end_position": "end",
                "chromosome_name": "chromosome",
            }
        )
        # use chr prefix for chromosome (cast to str first so Categorical /
        # ArrowExtensionArray returned by some pandas/scanpy backends does
        # not break the string concatenation; see also genomic_position_from_gtf)
        .assign(chromosome=lambda x: "chr" + x["chromosome"].astype(str))
    )

    gene_ids_adata = (adata.var_names if adata_gene_id is None else adata.var[adata_gene_id]).values
    missing_from_biomart = len(set(gene_ids_adata) - set(biomart_annot[biomart_gene_id].values))
    if missing_from_biomart:
        logging.warning(
            f"Biomart misses annotation for {missing_from_biomart} genes in adata. Did you use ENSEMBL ids?"
        )

    duplicated_symbols = np.sum(biomart_annot[biomart_gene_id].duplicated())
    if duplicated_symbols:
        logging.warning(f"Skipped {duplicated_symbols} genes because of duplicate identifiers in GTF file.")
        biomart_annot = biomart_annot.loc[~biomart_annot[biomart_gene_id].duplicated(keep=False), :]

    tmp_var = adata.var.copy()
    orig_index_name = tmp_var.index.name
    TMP_INDEX_NAME = "adata_var_index"
    tmp_var.index.name = TMP_INDEX_NAME
    tmp_var.reset_index(inplace=True)
    var_annotated = tmp_var.merge(
        biomart_annot,
        how="left",
        left_on=TMP_INDEX_NAME if adata_gene_id is None else adata_gene_id,
        right_on=biomart_gene_id,
        validate="one_to_one",
    )
    var_annotated.set_index(TMP_INDEX_NAME, inplace=True)
    var_annotated.index.name = orig_index_name

    if inplace:
        adata.var = var_annotated
    else:
        return var_annotated


def genomic_position_from_gtf(
    gtf_file: Path | str,
    adata: AnnData | None = None,
    *,
    gtf_gene_id: Literal["gene_id", "gene_name", "auto"] = "gene_name",
    adata_gene_id: str | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Get genomic gene positions from a GTF file.

    The GTF file needs to match the genome annotation used for your single cell dataset.

    .. warning::
        Currently only tested with GENCODE GTFs.

    Parameters
    ----------
    gtf_file
        Path to the GTF file
    adata
        Adds the genomic positions to `adata.var`. If adata is None, returns
        a data frame with the genomic positions instead.
    gtf_gene_id
        Use this GTF column to match it to anndata. ``"auto"`` first matches
        gene symbols via ``gene_name``, then fills unmatched Ensembl-style
        identifiers via ``gene_id``.
    adata_gene_id
        Match this column to the gene ids from the GTF file. Default: use
        adata.var_names.
    inplace
        If True, add the annotations directly to adata, otherwise return a dataframe.
    """
    try:
        import gtfparse
    except ImportError:
        raise ImportError(
            "genomic_position_from_gtf requires gtfparse as optional dependency. Please install it using `pip install gtfparse`."
        ) from None
    gtf = gtfparse.read_gtf(
        gtf_file, usecols=["seqname", "feature", "start", "end", "gene_id", "gene_name"]
    ).to_pandas()
    gtf = (
        gtf.loc[
            gtf["feature"] == "gene",
            ["seqname", "start", "end", "gene_id", "gene_name"],
        ]
        .drop_duplicates()
        .rename(columns={"seqname": "chromosome"})
    )
    # Categorical-safe: gtfparse (>=2.0 / pyarrow backend) returns Categorical /
    # ArrowExtensionArray columns. Downstream string ops (`"chr" + col`,
    # `.str.startswith`, `.str.replace`) all assume plain object/str dtype, so
    # cast string columns up-front. Keeps existing behavior on older gtfparse
    # versions that already returned object dtype.
    for _c in ("chromosome", "gene_id", "gene_name"):
        if _c in gtf.columns:
            gtf[_c] = gtf[_c].astype(str)
    # remove ensembl versions
    gtf["gene_id"] = gtf["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    chromosome_order = list(dict.fromkeys(gtf["chromosome"].dropna().astype(str)))

    tmp_var = adata.var.copy()
    orig_index_name = tmp_var.index.name
    gene_ids_adata = (
        pd.Series(adata.var_names, index=tmp_var.index, dtype="string")
        if adata_gene_id is None
        else tmp_var[adata_gene_id].astype("string")
    ).astype(str)
    gene_ids_adata_stripped = gene_ids_adata.str.replace(r"\.\d+$", "", regex=True)

    annot_cols = ["chromosome", "start", "end", "gene_id", "gene_name"]
    var_annotated = tmp_var.copy()
    for col in annot_cols:
        var_annotated[col] = pd.NA

    def _unique_gtf_map(key_col: Literal["gene_id", "gene_name"], keys: set[str]) -> pd.DataFrame:
        matched = gtf.loc[gtf[key_col].isin(keys), annot_cols].copy()
        duplicated_ids = matched[key_col].duplicated(keep=False)
        if duplicated_ids.any():
            logging.warning(
                f"Skipped {int(duplicated_ids.sum())} genes because of duplicate "
                f"{key_col} identifiers in GTF file."
            )
            matched = matched.loc[~duplicated_ids, :]
        return matched.set_index(key_col, drop=False)

    def _assign_annotations(mask: pd.Series, keys: pd.Series, gtf_map: pd.DataFrame) -> None:
        if not mask.any():
            return
        ann = gtf_map.loc[keys.loc[mask], annot_cols].copy()
        ann.index = var_annotated.index[mask]
        var_annotated.loc[mask, annot_cols] = ann

    if gtf_gene_id == "auto":
        gene_name_map = _unique_gtf_map("gene_name", set(gene_ids_adata))
        by_name = gene_ids_adata.isin(gene_name_map.index)
        _assign_annotations(by_name, gene_ids_adata, gene_name_map)

        remaining = ~by_name
        gene_id_map = _unique_gtf_map("gene_id", set(gene_ids_adata_stripped.loc[remaining]))
        by_gene_id = remaining & gene_ids_adata_stripped.isin(gene_id_map.index)
        _assign_annotations(by_gene_id, gene_ids_adata_stripped, gene_id_map)
        logging.info(
            "GTF auto annotation matched "
            f"{int(by_name.sum())} genes by gene_name and "
            f"{int(by_gene_id.sum())} genes by gene_id."
        )
    elif gtf_gene_id in {"gene_id", "gene_name"}:
        match_keys = gene_ids_adata_stripped if gtf_gene_id == "gene_id" else gene_ids_adata
        gtf_map = _unique_gtf_map(gtf_gene_id, set(match_keys))
        matched = match_keys.isin(gtf_map.index)
        _assign_annotations(matched, match_keys, gtf_map)
    else:
        raise ValueError("gtf_gene_id must be one of 'gene_id', 'gene_name', or 'auto'.")

    missing_from_gtf = int(var_annotated["chromosome"].isna().sum())
    if missing_from_gtf:
        logging.warning(f"GTF file misses annotation for {missing_from_gtf} genes in adata.")

    # Keep coordinate columns numeric after auto assignment. Leaving pd.NA in an
    # object column makes AnnData/HDF5 try to serialize mixed objects as strings.
    var_annotated["start"] = pd.to_numeric(var_annotated["start"], errors="coerce")
    var_annotated["end"] = pd.to_numeric(var_annotated["end"], errors="coerce")
    var_annotated.index.name = orig_index_name

    # if not a gencode GTF, let's add 'chr' prefix:
    if np.all(~var_annotated["chromosome"].dropna().str.startswith("chr")):
        var_annotated["chromosome"] = "chr" + var_annotated["chromosome"]
        chromosome_order = [f"chr{c}" for c in chromosome_order]

    present_chromosomes = var_annotated["chromosome"].dropna().astype(str).unique().tolist()
    chromosome_categories = chromosome_order + [
        c for c in present_chromosomes if c not in set(chromosome_order)
    ]
    var_annotated["chromosome"] = pd.Categorical(
        var_annotated["chromosome"],
        categories=chromosome_categories,
        ordered=True,
    )

    if inplace:
        adata.var = var_annotated
    else:
        return var_annotated
