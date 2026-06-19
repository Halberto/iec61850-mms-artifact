# Limitations

This Git repository does not include the full unprocessed raw capture files, most large processed tables, or the full set of experiment checkpoints. Those artifacts are better published as release archives or DOI-backed deposits.

The normalized corpus files (`mms_capture_normalized.csv.gz`, `mms_capture_normalized.jsonl.gz`) are derived from the original raw capture by removing structural redundancy and merging reviewed labels. The following information from the original capture is not present in the normalized files:

- Full per-layer protocol dissection (TPKT, COTP, Session, Presentation layer detail)
- Response-side raw MMS hex bytes (`response_raw_mms_hex`)
- Response packet dissection (`response_dissection`)

These fields are available in the original raw capture CSV, which is retained as a private archive and can be regenerated using `src/utils/normalize_dataset.py` with different column selections if needed.

The repository is intended to be intentional, reviewable, and reproducible — not a mirror of the original working directory.

## Attack-class scarcity

The captured attack subset is small by nature: **105 real malicious packets** (all IEDExplorer-origin IEC 61850 report PDUs), spanning **2 IEDs** and **7 controllable switch objects** (of which `Q8CSWI1` is the earth switch). This limits the statistical power of purely supervised detection trained on real positives alone, and is reported as a limitation of the dataset rather than mitigated through synthetic data generation.
