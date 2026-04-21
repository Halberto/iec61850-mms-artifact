"""
Normalize the raw MMS capture CSV into a compact, deduplicated JSONL.

Each record is flattened so that:
- Nested JSON fields (dissection, control_parameters, labels) are
  expanded into top-level scalar fields.
- Fields inside `dissection` that duplicate top-level scalars are dropped.
- `control_variables` is omitted when it is identical to `variables`.
- `control_parameters` typed objects are replaced by direct scalar fields.
- Response-side heavy blobs (response_dissection, response_raw_mms_hex)
  are dropped; lightweight response scalars are kept.
- `raw_mms_hex` is kept by default (pass --drop-hex to omit it).

What is dropped:
  dissection.ipv4.*               -> already in src_ip / dst_ip
  dissection.transport_summary.*  -> already in direction / origin / timestamp
  dissection.frame.timestamp      -> already in wtimestamp
  dissection.tpkt / cotp / session / presentation -> protocol boilerplate
  dissection.mms_pdu_details.raw_hex_first_32 -> prefix of raw_mms_hex
  control_parameters (typed object) -> replaced by flat ctl_* scalars
  control_variables               -> dropped when equal to variables
  response_dissection             -> large blob, not useful without re-capture
  response_raw_mms_hex            -> large hex, in sidecar if needed

What is added (unique fields extracted from dissection):
  frame_number, frame_mms_size
  src_mac, dst_mac
  prp_sequence, prp_lan_id, prp_lsdu_size
  ipv4_ttl
  tcp_seq, tcp_ack, tcp_flags, tcp_window
  mms_payload_size, mms_first_tag

What is added (from control_parameters):
  origin_identifier, origin_category, ctl_num, ctl_timestamp, ctl_test, ctl_check

What is added (from control_value):
  ctl_val_type, ctl_val

What is added (from labels):
  label_off_hours, label_unexpected_target,
  label_sequence_violation, label_abnormal_rate, label_new_connection
"""

import argparse
import csv
import gzip
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

# Scalar CSV columns to pass through unchanged
PASSTHROUGH_SCALARS = [
    "direction",
    "src_ip", "src_port",
    "dst_ip", "dst_port",
    "frame_len",
    "origin",
    "stream_id",
    "session_id",
    "pdu_type", "service", "invoke_id",
    "summary",
    "response_pdu_type", "response_service",
    "response_timestamp",
    "response_src_ip", "response_src_port",
    "response_dst_ip", "response_dst_port",
    "response_origin",
    "variable_list_name",
    "control_object", "control_action", "control_field",
    "is_control", "control_op_count",
    "new_connection",
    "result",      # success / failure for CONFIRMED_RESPONSE rows
    "error_code",  # detail when service returns an error
]

# CSV columns that contain JSON arrays or objects to keep as structured values
PASSTHROUGH_JSON = ["variables", "data", "access_result"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize the raw MMS CSV into a compact, deduplicated CSV and JSONL."
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        required=True,
        help="Path to the raw MMS CSV export to normalize.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(RAW_DATA_DIR / "mms_capture_normalized.jsonl.gz"),
        help="Output path for the normalized JSONL (or .jsonl for uncompressed).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(RAW_DATA_DIR / "mms_capture_normalized.csv.gz"),
        help="Output path for the normalized CSV (or .csv for uncompressed). "
             "Array fields are JSON-encoded strings in CSV format.",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Write plain files instead of gzip-compressed output.",
    )
    parser.add_argument(
        "--drop-hex",
        action="store_true",
        help="Omit raw_mms_hex from the output (saves space; loses byte-level provenance).",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Path to the supervised labels CSV (e.g. data/labels/mms_full_capture_supervised_labels.csv). "
             "When provided, label columns are merged into every output row by line_number.",
    )
    return parser


def safe_json(value: str | None) -> object:
    """Parse a JSON string. Returns None on failure or empty input."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            return None


def extract_dissection_unique(raw: str) -> dict:
    """
    Extract only the fields from the `dissection` object that are NOT
    already present at the top level of the CSV row.
    """
    d = safe_json(raw)
    if not isinstance(d, dict):
        return {}

    out: dict = {}

    frame = d.get("frame") or {}
    if frame.get("number") is not None:
        out["frame_number"] = frame["number"]
    # frame.size = MMS payload size (differs from top-level frame_len = Ethernet frame)
    if frame.get("size") is not None:
        out["frame_mms_size"] = frame["size"]

    eth = d.get("ethernet") or {}
    if eth.get("source_mac"):
        out["src_mac"] = eth["source_mac"]
    if eth.get("destination_mac"):
        out["dst_mac"] = eth["destination_mac"]

    prp = d.get("prp") or {}
    if prp.get("sequence") is not None:
        out["prp_sequence"] = prp["sequence"]
        out["prp_lan_id"] = prp.get("lan_id")
        out["prp_lsdu_size"] = prp.get("lsdu_size")

    # ipv4.ttl is unique; source_ip / destination_ip are already at top level
    ipv4 = d.get("ipv4") or {}
    if ipv4.get("ttl") is not None:
        out["ipv4_ttl"] = ipv4["ttl"]

    tcp = d.get("tcp") or {}
    if tcp:
        if tcp.get("seq") is not None:
            out["tcp_seq"] = tcp["seq"]
        if tcp.get("ack") is not None:
            out["tcp_ack"] = tcp["ack"]
        if tcp.get("flags"):
            out["tcp_flags"] = tcp["flags"]
        if tcp.get("window") is not None:
            out["tcp_window"] = tcp["window"]

    mms = d.get("mms_pdu_details") or {}
    # raw_hex_first_32 is just a prefix of raw_mms_hex — skip it
    if mms.get("total_payload_size") is not None:
        out["mms_payload_size"] = mms["total_payload_size"]
    if mms.get("first_tag"):
        out["mms_first_tag"] = mms["first_tag"]

    # tpkt / cotp / session / presentation are protocol boilerplate — skipped

    return {k: v for k, v in out.items() if v is not None}


def extract_control_parameters(raw: str) -> dict:
    """
    Flatten the typed control_parameters object into direct scalar fields.
    Each typed object looks like {"type": "...", "value": ...}.
    """
    d = safe_json(raw)
    if not isinstance(d, dict):
        return {}

    def get_val(key: str) -> object:
        entry = d.get(key) or {}
        if isinstance(entry, dict):
            return entry.get("value")
        return None

    out: dict = {}
    oi = get_val("originIdentifier")
    if oi is not None:
        out["origin_identifier"] = oi
    oc = get_val("originCategory")
    if oc is not None:
        out["origin_category"] = oc
    cn = get_val("ctlNum")
    if cn is not None:
        out["ctl_num"] = cn
    ts = get_val("timestamp")
    if ts is not None:
        out["ctl_timestamp"] = ts
    test_val = get_val("test")
    if test_val is not None:
        out["ctl_test"] = test_val
    check_val = get_val("check")
    if check_val is not None:
        out["ctl_check"] = check_val
    return out


def extract_control_value(raw: str) -> dict:
    """
    Flatten {"type": "boolean", "value": false} → ctl_val_type, ctl_val.
    """
    d = safe_json(raw)
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    if d.get("type") is not None:
        out["ctl_val_type"] = d["type"]
    if d.get("value") is not None:
        out["ctl_val"] = d["value"]
    return out


def open_writer(path: Path, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


# Label columns to pull from the supervised labels CSV.
# Excludes: line_number (join key), timestamp (already in event), stream_id (duplicate).
LABEL_COLUMNS = [
    "final_tag",
    "seed_is_attack",
    "supervised_is_anomaly",
    "scenario_id",
    "scenario_role",
    "scenario_seed_count",
    "scenario_member_count",
    "scenario_group_key",
    "scenario_window_start_line",
    "scenario_window_end_line",
    "scenario_core_start_line",
    "scenario_core_end_line",
    "scenario_start_timestamp",
    "scenario_end_timestamp",
    "scenario_duration_seconds",
]


def load_labels(path: Path) -> dict[int, dict]:
    """Load the supervised labels CSV into a dict keyed by line_number (int)."""
    labels: dict[int, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = int(row["line_number"])
            except (KeyError, ValueError):
                continue
            labels[key] = {col: row.get(col, "") for col in LABEL_COLUMNS}
    return labels


# Fields whose values are lists/dicts in JSONL but are JSON-encoded strings in CSV
ARRAY_FIELDS = {"variables", "control_variables", "data", "access_result"}


def record_to_csv_row(record: dict) -> dict:
    """Convert a normalized record dict to a flat CSV-safe dict.
    Array/object fields are serialized as compact JSON strings.
    """
    out = {}
    for k, v in record.items():
        if k in ARRAY_FIELDS and isinstance(v, (list, dict)):
            out[k] = json.dumps(v, separators=(",", ":"))
        else:
            out[k] = v
    return out


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    compress = not args.no_gzip

    # Load ground-truth labels if provided
    label_map: dict[int, dict] = {}
    if args.labels_csv:
        labels_path = Path(args.labels_csv)
        print(f"Loading labels from {labels_path} ...")
        label_map = load_labels(labels_path)
        print(f"  Loaded {len(label_map):,} label rows")

    row_count = 0
    csv_fieldnames: list[str] | None = None
    csv_writer: csv.DictWriter | None = None

    with input_csv.open("r", encoding="utf-8", newline="") as src, \
         open_writer(output_jsonl, compress) as jsonl_dst, \
         open_writer(output_csv, compress) as csv_dst:

        reader = csv.DictReader(src)

        for row_count, row in enumerate(reader, start=1):
            record: dict = {"line_number": row_count}

            # Canonical timestamp (wtimestamp in CSV, timestamp when already renamed)
            ts = row.get("wtimestamp") or row.get("timestamp") or ""
            if ts:
                record["timestamp"] = ts

            # Pass-through scalar fields
            for field in PASSTHROUGH_SCALARS:
                val = row.get(field, "")
                if val not in ("", None):
                    record[field] = val

            # Pass-through JSON array / object fields
            for field in PASSTHROUGH_JSON:
                parsed = safe_json(row.get(field, ""))
                if parsed is not None:
                    record[field] = parsed

            # control_value → flat scalars
            record.update(extract_control_value(row.get("control_value", "")))

            # control_parameters → flat scalars (replaces the typed nested object)
            record.update(extract_control_parameters(row.get("control_parameters", "")))

            # variables vs control_variables: keep variables;
            # only add control_variables when the two differ
            vars_parsed = safe_json(row.get("variables", ""))
            cvars_parsed = safe_json(row.get("control_variables", ""))
            if vars_parsed is not None:
                record["variables"] = vars_parsed
            if cvars_parsed is not None and cvars_parsed != vars_parsed:
                record["control_variables"] = cvars_parsed

            # dissection → keep only unique fields, drop all redundant ones
            record.update(extract_dissection_unique(row.get("dissection", "")))

            # NOTE: labels from the raw CSV are inaccurate placeholders.
            # Ground-truth labels are joined here from --labels-csv by line_number.
            if label_map:
                lbl = label_map.get(row_count, {})
                for col in LABEL_COLUMNS:
                    val = lbl.get(col, "")
                    if val not in ("", None):
                        record[col] = val

            # raw_mms_hex (optional; omit with --drop-hex)
            if not args.drop_hex:
                hex_val = row.get("raw_mms_hex", "")
                if hex_val:
                    record["raw_mms_hex"] = hex_val

            # --- JSONL output ---
            jsonl_dst.write(json.dumps(record, separators=(",", ":")) + "\n")

            # --- CSV output ---
            csv_row = record_to_csv_row(record)
            if csv_writer is None:
                # Initialise header from first record's keys
                csv_fieldnames = list(csv_row.keys())
                csv_writer = csv.DictWriter(
                    csv_dst,
                    fieldnames=csv_fieldnames,
                    extrasaction="ignore",
                    restval="",
                )
                csv_writer.writeheader()
            csv_writer.writerow(csv_row)

    print(f"Normalized {row_count:,} rows")
    print(f"JSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")


if __name__ == "__main__":
    main()
