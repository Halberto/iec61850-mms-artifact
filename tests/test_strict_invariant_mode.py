#!/usr/bin/env python3
"""Focused test for the strict-deterministic-invariants mode of hybrid_mms_ids.

It exercises the two recovery paths that the default (online-learned) mode misses,
without needing the full 428k-row capture:

  * Warm-up window: a hard protocol violation that lands inside the statistical
    warm-up prefix is dropped by the default mode but must be emitted in strict
    mode (because the baseline is provisioned and protocol rules score from row 1).
  * De-duplication: a second identical hard violation inside the suppression
    window is suppressed by the default mode but must be emitted in strict mode
    (each hard violation is a distinct protocol event, not statistical noise).

A legitimate post-warm-up control WRITE is included so the default mode can also
detect the post-warm-up attack as likely-attack (online learning populates the
authorized-origin set), demonstrating strict mode adds recall without regressing
the case the default mode already handles.

Run: python tests/test_strict_invariant_mode.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "baseline"))

import hybrid_mms_ids as ids  # noqa: E402

# A faithful report PDU access_result: a CSWI control reference (the control-context
# hint), plus a structure carrying the origin octet-string and the ctlNum unsigned.
ATTACK_ORIGIN = "IEDEXPLORER_ORIGIN"
LEGIT_ORIGIN = "SAT200_ORIGIN"


def _attack_report_access_result() -> str:
    return json.dumps([
        {"type": "visible-string", "value": "E03A103_CTRL/LLN0$RP$A_URCB_10"},
        {"type": "structure", "value": [
            {"type": "octet-string", "value": ATTACK_ORIGIN},
            {"type": "unsigned", "value": 0},
        ]},
        {"type": "visible-string", "value": "E03A103_CTRL/Q1CSWI1$ST$Pos"},
    ])


def _report_row(line_no: int, ts: str) -> dict:
    return {
        "line_number": line_no,
        "wtimestamp": ts,
        "direction": "RESPONSE",
        "service": "UNCONFIRMED",
        "src_ip": "10.0.19.49",
        "dst_ip": "10.0.19.39",
        "stream_id": "10.0.19.39:56638-10.0.19.49:102",
        "variable_list_name": "RPT",
        "summary": "<RPT>",
        "control_object": "",
        "control_action": "",
        "invoke_id": "",
        "is_control": "0",
        "raw_mms_hex": "",
        "control_parameters": "{}",
        "variables": "",
        "access_result": _attack_report_access_result(),
    }


def _legit_write_row(line_no: int, ts: str) -> dict:
    return {
        "line_number": line_no,
        "wtimestamp": ts,
        "direction": "REQUEST",
        "service": "WRITE",
        "src_ip": "10.0.19.39",
        "dst_ip": "10.0.19.49",
        "stream_id": "10.0.19.39:50000-10.0.19.49:102",
        "variable_list_name": "",
        "summary": "legit control write",
        "control_object": "Q1CSWI1$CO$Pos",
        "control_action": "",
        "invoke_id": "100",
        "is_control": "1",
        "raw_mms_hex": "",
        "control_parameters": json.dumps({
            "originIdentifier": {"value": LEGIT_ORIGIN},
            "ctlNum": {"value": 5},
        }),
        "variables": "",
        "access_result": "",
    }


CSV_COLUMNS = [
    "line_number", "wtimestamp", "direction", "service", "src_ip", "dst_ip",
    "stream_id", "variable_list_name", "summary", "control_object",
    "control_action", "invoke_id", "is_control", "raw_mms_hex",
    "control_parameters", "variables", "access_result",
]

# Warm-up ends at 01:44:34; row A is inside it, W/B/C are after it.
ROW_A_WARMUP = ("422", "2026-02-14T01:42:29.429905")   # attack inside warm-up
ROW_W_WRITE = ("900", "2026-02-14T01:44:40.000000")    # legit write, post warm-up
ROW_B_ATTACK = ("165400", "2026-02-14T01:45:00.000000")  # attack, post warm-up
ROW_C_DUP = ("165404", "2026-02-14T01:45:00.500000")     # identical attack, +0.5s
TRAIN_END = "2026-02-14T01:44:34.309070"


def _write_input(path: Path) -> None:
    rows = [
        _report_row(int(ROW_A_WARMUP[0]), ROW_A_WARMUP[1]),
        _legit_write_row(int(ROW_W_WRITE[0]), ROW_W_WRITE[1]),
        _report_row(int(ROW_B_ATTACK[0]), ROW_B_ATTACK[1]),
        _report_row(int(ROW_C_DUP[0]), ROW_C_DUP[1]),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _make_args(**overrides):
    saved = sys.argv
    sys.argv = ["hybrid_mms_ids.py"]
    try:
        args = ids.parse_args()
    finally:
        sys.argv = saved
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _model() -> dict:
    return {
        "known_origins": [LEGIT_ORIGIN],
        "allowed_origin": LEGIT_ORIGIN,
        "write_ctlnums_by_origin": {LEGIT_ORIGIN: [5]},
        "train_end": TRAIN_END,
        "window_sec": 30.0,
    }


def _run(strict: bool, tmp: Path) -> list[dict]:
    in_csv = tmp / "input.csv"
    out_csv = tmp / ("alerts_strict.csv" if strict else "alerts_default.csv")
    _write_input(in_csv)
    args = _make_args(strict_deterministic_invariants=strict, suppress_sec=30.0)
    ids.detect_with_hybrid_ids(in_csv, out_csv, _model(), args)
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _emitted_lines(rows: list[dict], tag: str | None = None) -> set[str]:
    return {
        r["line_number"]
        for r in rows
        if r.get("event_source") == "packet" and (tag is None or r["final_tag"] == tag)
    }


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        default_rows = _run(strict=False, tmp=tmp)
        strict_rows = _run(strict=True, tmp=tmp)

    # The detector numbers output rows by their position in the input stream
    # (enumerate(start=1)), independent of any "line_number" column. Rows are written
    # in the order A, W, B, C, so they surface as detector lines "1", "2", "3", "4".
    a, w, b, c = "1", "2", "3", "4"

    default_attacks = _emitted_lines(default_rows, "likely-attack")
    strict_attacks = _emitted_lines(strict_rows, "likely-attack")

    print(f"default likely-attack lines: {sorted(default_attacks)}")
    print(f"strict  likely-attack lines: {sorted(strict_attacks)}")

    # --- Default mode expectations -------------------------------------------------
    if a in default_attacks:
        failures.append("DEFAULT: warm-up attack (A) should be dropped, but was emitted")
    if b not in default_attacks:
        failures.append("DEFAULT: post-warm-up attack (B) should be detected as likely-attack")
    if c in default_attacks:
        failures.append("DEFAULT: duplicate attack (C) should be suppressed, but was emitted")
    if w in _emitted_lines(default_rows):
        failures.append("DEFAULT: legitimate control write (W) should not raise an alert")

    # --- Strict mode expectations --------------------------------------------------
    for label, line in (("warm-up A", a), ("post-warm-up B", b), ("duplicate C", c)):
        if line not in strict_attacks:
            failures.append(f"STRICT: attack {label} (line {line}) should be emitted as likely-attack")
    if w in _emitted_lines(strict_rows):
        failures.append("STRICT: legitimate control write (W) should not raise an alert")

    # --- No-regression cross-check -------------------------------------------------
    if not default_attacks.issubset(strict_attacks):
        failures.append("STRICT should be a superset of DEFAULT detections (no regression)")

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPASS: strict mode recovers warm-up (A) and de-duplicated (C) attacks; "
          "default mode behaviour and clean-write handling unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
