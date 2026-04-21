# Submission Artifact Guide

## What the manuscript should point to

Do not point reviewers to the repository root or a moving branch.

Use immutable targets instead:

- tagged release landing page: `https://github.com/<owner>/<repo>/releases/tag/<tag>`
- detector source at the tagged release: `https://github.com/<owner>/<repo>/blob/<tag>/src/baseline/hybrid_mms_ids.py`
- dataset archive attached to the tagged release: `https://github.com/<owner>/<repo>/releases/download/<tag>/mms-dataset-<tag>.zip`
- optional models and results archive: `https://github.com/<owner>/<repo>/releases/download/<tag>/mms-models-results-<tag>.zip`

If you archive the dataset through Zenodo or a similar service, replace the dataset archive URL with the DOI landing page for the archived release.

## Manuscript-ready wording

Suggested artifact paragraph:

"We release an immutable artifact package for the MMS IDS at `<release-url>`. The full detector implementation is included in `src/baseline/hybrid_mms_ids.py` within the tagged source release. The full-capture dataset is available as `data/raw/mms_capture_normalized.csv.gz` (20 MB) or in JSONL format as `data/raw/mms_capture_normalized.jsonl.gz` (63 MB), or through an archival DOI mirror, so that reviewers access a fixed artifact rather than a moving repository branch."

Suggested dataset sentence:

"The released dataset contains the normalized full-corpus MMS capture (428,204 records with ground-truth labels), reviewed scenario labels, and sample workflow assets."

Suggested code sentence:

"The released source contains the hybrid rule-plus-statistical detector, preprocessing scripts, feature generation pipeline, and baseline training code."

## Why some sample files are excluded from Git

The repository is self-contained for the normalized corpus workflow. Two large sample files are excluded from the Git tree because they exceed safe size limits:

- `data/sample/mms_sample_100k.csv` at about 417 MB
- `data/sample/mms_sample_100k.jsonl` at about 425 MB

These are listed in `data/metadata/release_inventory.csv` with `recommended_publication: release-archive-or-doi` and should be distributed as release assets or via an archival service. The original unprocessed raw capture CSV (~1.72 GB) is kept as a private archive and is not published. See `data/metadata/release_inventory.csv` for the full manifest.

## Recommended release contents

- Code and normalized dataset (this repository): `README.md`, `docs/`, `src/`, `data/raw/`, `data/labels/`, `data/sample/` (small files only), `results/expected_outputs/`, `configs/`, `requirements.txt`, `CITATION.cff`, `LICENSE`
- Large sample archive (release asset or DOI): `data/sample/mms_sample_100k.csv`, `data/sample/mms_sample_100k.jsonl`

## Pre-submission checklist

- Create a repository and push the contents of this repository.
- Verify that `data/sample/mms_sample_100k.csv` and `data/sample/mms_sample_100k.jsonl` are listed in `.gitignore` and not committed.
- Create a tag such as `v1.0.0-artifact`.
- Attach a large-sample archive to that tag, or mint a DOI for it.
- Update the manuscript so the detector claim points to `src/baseline/hybrid_mms_ids.py` in the tagged release.
- Update the manuscript so the dataset claim points to the normalized corpus files in `data/raw/` or the versioned archive.
- Verify that the final public links resolve without requiring branch navigation.
