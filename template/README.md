# cnvturbo Templates

This folder contains runnable examples for the recommended R inferCNV-compatible
workflow. The scripts are intentionally conservative and explicit so users can
adapt them to their own AnnData files without depending on R inferCNV outputs.

## Scripts

- `01_r_compatible_analysis.py`  
  Runs one sample through GTF annotation, R-compatible preprocessing, hspike
  emission calibration, subcluster HMM calling, and strict tumor calling.

- `02_visualization.py`  
  Reads the `.h5ad` files written by script 01 and generates common diagnostic
  plots.

- `03_benchmark_against_infercnv.py`  
  Optional validation script for users who already have R inferCNV-derived
  per-cell calls. It compares cnvturbo calls and continuous scores against a
  CSV exported from R.

## Expected AnnData Input

Each input `.h5ad` should contain:

- raw integer counts in `adata.layers["counts"]` or `adata.X`;
- a reference annotation column in `adata.obs`, for example `cell_type`;
- gene identifiers in `adata.var_names` or a gene-id column;
- no precomputed R inferCNV results are required.

If gene coordinates are missing, script 01 can inject them from a GTF with
`cnvturbo.io.genomic_position_from_gtf(..., gtf_gene_id="auto")`.

## Minimal Run

```bash
cd /path/to/project
python cnvturbo/template/01_r_compatible_analysis.py
python cnvturbo/template/02_visualization.py
```

Edit the configuration block near the top of each script before running:

- `INPUT_H5AD`
- `GTF_FILE`
- `REFERENCE_KEY`
- `REFERENCE_CATEGORIES`
- `OUTPUT_DIR`

For strict R-equivalent tumor calling, script 01 uses:

```text
cnv_score > P95(cnv_score | reference cells)
AND
proportion_cnv > P95(proportion_cnv | reference cells)
```

where `cnv_score` is computed from the R-compatible denoised copy-ratio matrix,
and `proportion_cnv` is the HMM non-neutral state fraction written by
`hmm_call_subclusters`.

## Acknowledgements

`cnvturbo` stands on the shoulders of two important open-source CNV projects:

- [broadinstitute/infercnv](https://github.com/broadinstitute/infercnv)  
  The R inferCNV project defines the reference workflow that this template aims
  to reproduce in a standalone Python implementation.

- [icbi-lab/infercnvpy](https://github.com/icbi-lab/infercnvpy)  
  The original Python/Scanpy-oriented inferCNV implementation inspired the
  AnnData-friendly API surface and provided part of the historical foundation
  from which `cnvturbo` evolved.

We gratefully acknowledge the authors and contributors of both projects.
