# Reproduction Materials

These materials reproduce the analyses for "Rapid sequential touches complicate whisking-phase analysis in mouse active touch."

## Contents

- `manifest.json`: versioned DANDI asset manifest for DANDI:000013, version 0.220126.2143.
- `data/touch_events.csv`: event table with reconstructed spike windows, interval measures, layer metadata, and continuous kinematic or contact variables.
- `scripts/download_data.py`: optional downloader and checksum verifier for the 52 source NWB files.
- `scripts/extract_features.py`: reconstructs event variables from the NWB files and validates spike counts.
- `scripts/run_analysis.py`: applies the layer 4 eligibility criteria, runs all models and sensitivity analyses, and generates source-data files and figures.
- `validation/extraction_validation.json`: file-level extraction and checksum report.
- `results/`: numerical outputs reported in the manuscript and supporting information, including observed-neighbor isolation, crossed baseline/response-window estimates, phase-branch contrasts, recording-location summaries, contact-duration summaries, and label-level coefficient files.
- `figures/`: the two manuscript figures in PNG format.
- `requirements.txt`: tested Python package versions.
- `checksums.sha256`: SHA-256 checksums for every archived file except the checksum file itself.

Raw NWB files are omitted because they require approximately 11 GB. The manifest and downloader retrieve and verify the exact public files.

## Reproduce from the included event table

Create a Python environment with the versions in `requirements.txt`, then run:

```bash
python3 scripts/run_analysis.py data/touch_events.csv reproduced
```

Expected top-level counts are 19,675 eligible layer 4 touches, 41 recording files, and 21 archive subject labels. The complete release contains 23 subject labels, whereas the source article reports 21 mice across 52 recording sessions. No crosswalk verifies a one-to-one relation, so the labels are used as grouping keys rather than as a verified animal census. The primary set with at least 50 ms to both neighboring onsets contains 12,773 touches.

## Reconstruct the event table from NWB

Download the frozen source files:

```bash
python scripts/download_data.py manifest.json raw_nwb --workers 4
```

Reconstruct the event table from the source NWB files and the included original event index:

```bash
python scripts/extract_features.py data/source_touch_index.csv raw_nwb data/touch_events_reconstructed.csv validation/extraction_validation_reconstructed.json
```

Then run the analysis on `data/touch_events_reconstructed.csv`. The extraction check permits a one-spike difference only at explicit half-open window boundaries, reflecting inconsistent endpoint inclusion in the earlier processed count fields.

## Analysis conventions

The manuscript analysis retains layer 4 touches during the first 2 s after pole entry, excludes optogenetic-stimulation trials and missing phase, and uses archive subject labels for final aggregation. Cross-validation holds out complete trials. Phase is represented with first- and second-harmonic sine/cosine terms in the primary model. Exact sign tests complement signed-rank tests for the primary phase comparisons, and penalty sensitivity covers 0.01, 0.1, 1, and 10. Peak post-contact curvature and touch duration are analyzed in a separate coefficient model and are excluded from the pre-touch predictive adjustment because they occur during the response interval. A descriptive sensitivity adds these post-onset measures to the predictive adjustment. Additional outputs report direct phase-branch contrasts, C2 and surrounding-barrel summaries, and a stricter condition requiring observed within-trial neighbors on both sides.

## Terms of use

Code is distributed under the MIT License. Derived tabular data and documentation are distributed under the Creative Commons Attribution 4.0 International License, consistent with the source DANDI dataset. See `LICENSE_CODE.txt`, `LICENSE_DATA.txt`, and `TERMS_OF_USE.md`.
