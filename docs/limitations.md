# Limitations

This Git repository does not include the full unprocessed raw capture files, most large processed tables, or the full set of experiment checkpoints. Those artifacts are better published as release archives or DOI-backed deposits.

The normalized corpus files (`mms_capture_normalized.csv.gz`, `mms_capture_normalized.jsonl.gz`) are derived from the original raw capture by removing structural redundancy and merging reviewed labels. The following information from the original capture is not present in the normalized files:

- Full per-layer protocol dissection (TPKT, COTP, Session, Presentation layer detail)
- Response-side raw MMS hex bytes (`response_raw_mms_hex`)
- Response packet dissection (`response_dissection`)

These fields are available in the original raw capture CSV, which is retained as a private archive and can be regenerated using `src/utils/normalize_dataset.py` with different column selections if needed.

The repository is intended to be intentional, reviewable, and reproducible — not a mirror of the original working directory.
