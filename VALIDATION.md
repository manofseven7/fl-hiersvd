# Validation and correction record

This package was reviewed against the experimental protocol and communication
accounting stated in the manuscript.

## Corrections in version 1.1.0

- Restored the article's one-hidden-layer, 128-unit tabular MLP.
- Counted complete three-link traffic in MiB, including one downlink delivery
  per participating client.
- Made UCI HAR feature selection default to on only for the Proposed method.
- Computed model-guided feature importance from the final warm-up local update;
  the earlier implementation inadvertently subtracted a tensor from itself.
- Corrected PyTorch input-feature orientation (features are columns of a
  `Linear.weight` tensor) and implemented actual compact-coordinate SVD payloads.
- Added Macro-F1 to every trajectory and to the table/figure scripts.
- Matched baseline link policies to Table 3: QSGD/FedCOM/Top-k use compressed
  client uplinks and dense remaining links; PowerSGD uses factor uplinks and a
  dense global downlink; SignSGD counts packed bits.
- Replaced the SVD stand-in in PowerSGD with a warm-started power iteration and
  client error feedback, including intermediate-P broadcast accounting.
- Corrected SCAFFOLD option-II client/server control updates and FedNova step
  normalisation.
- Implemented persistent bidirectional teacher/student training for the
  disclosed FedKD-style adaptation.
- Fixed Fashion-MNIST vectorised normalisation, empty-client handling, CUDA
  tensor placement, table communication means, plot tag selection, and missing
  LICENSE/test files.
- Added executable training-only feature-retention selection, fresh-process
  timing, and scalability sweep scripts for the auxiliary manuscript tables.

## Verification status

- Python syntax compilation and compression-level smoke checks pass in the
  review environment.
- `tests/test_core.py` covers compact masking, feature scores, the two published
  dense-communication endpoints, and one synthetic round for every method.
- A complete five-seed, 30/400-round reproduction was not rerun during this code
  review. Run `pytest -q`, then `bash scripts/run_all.sh`, before replacing any
  archived article result. Full runs are the numerical verification; smoke
  tests only verify code paths and invariants.
