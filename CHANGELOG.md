# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-24

### Added
- Initial public release of `cnvturbo`.
- `tl.infercnv_r_compat`: gene-space, copy-ratio R inferCNV-equivalent 8-step pipeline.
- `tl.compute_hspike_emission_params`: hspike-based HMM emission calibration.
- `tl.hmm_call_subclusters`: cell-level Tumor / Normal classification via per-subcluster HMM i6 (R-equivalent Viterbi + denoise).
- `tl.hmm_call_cells`: cell-level HMM caller without subclustering.
- Numba (CPU) and PyTorch (GPU) accelerated sliding-window convolution and batched Viterbi.
- Continuous cell-level CNV burden score `cnv_call_score`.
- Verified 100% cell-level concordance with R inferCNV's HMM output on 3 PDAC samples (15,135 cells), at 100–230× speed-up.

[0.1.0]: https://github.com/LogicByteCraft/cnvturbo/releases/tag/v0.1.0

