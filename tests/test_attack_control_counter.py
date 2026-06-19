#!/usr/bin/env python3
"""Regression test for sequential attack report payload ctlNum mutation.

Run: python tests/test_attack_control_counter.py
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "augmentation"))

from attack_model import AttackGenerativeModel  # noqa: E402


ACCESS_RESULT = json.dumps([
    {"type": "visible-string", "value": "E03A103_CTRL/LLN0$RP$A_URCB_10"},
    {"type": "unsigned", "value": 160},
    {"type": "visible-string", "value": "E03A103_CTRL/Q1CSWI1$ST$Pos"},
    {"type": "structure", "value": [
        {"type": "structure", "value": [
            {"type": "integer", "value": 2},
            {"type": "octet-string", "value": "IEDEXPLORER"},
        ]},
        {"type": "unsigned", "value": 14},
        {"type": "utc-time", "value": "2026-02-14T00:42:28"},
    ]},
])


def _unsigned_values(node, out):
    if isinstance(node, list):
        for item in node:
            _unsigned_values(item, out)
    elif isinstance(node, dict):
        if node.get("type") == "unsigned":
            out.append(node.get("value"))
        _unsigned_values(node.get("value"), out)


def main() -> int:
    mutated = AttackGenerativeModel._mutate_access_result(
        ACCESS_RESULT,
        ["Q0CSWI1"],
        new_seq=999,
        new_clock=datetime(2026, 2, 14, 1, 2, 3),
        new_ctl_num=1,
    )
    parsed = json.loads(mutated)
    unsigneds = []
    _unsigned_values(parsed, unsigneds)

    if unsigneds != [999, 1]:
        print(f"FAIL: expected report seq 999 and payload ctlNum 1, got {unsigneds}")
        return 1
    if "Q0CSWI1$ST$Pos" not in mutated:
        print("FAIL: object token mutation did not preserve the report structure")
        return 1

    profile = {
        "constants": {
            "tag": "attack",
            "direction": "RESPONSE",
            "service": "UNCONFIRMED",
            "variable_list_name": "RPT",
            "octet_identities": "IEDEXPLORER",
            "dst_ip": "10.0.19.39",
            "origin_category": "",
        },
        "src_ip_pmf": {"10.0.19.49": 1},
        "ied_objects": {"E03A103_CTRL": {"CSWI": {"Q1CSWI1": 1}, "CILO": {}}},
        "ied_urcb": {},
        "object_markov_start": {},
        "object_markov": {},
        "stream_seq": {"stream-a": {"start_min": 10, "increment_pmf": {"1": 1}}},
        "stream_timing": {
            "stream-a": {
                "start_timestamp": "2026-02-14T00:00:00",
                "gap_seconds_sample": [0.1],
            }
        },
    }
    template = {
        "src_ip": "10.0.19.49",
        "dst_ip": "10.0.19.39",
        "stream_id": "stream-a",
        "tag": "attack",
        "direction": "RESPONSE",
        "service": "UNCONFIRMED",
        "variable_list_name": "RPT",
        "octet_identities": "IEDEXPLORER",
        "access_result": ACCESS_RESULT,
    }
    model = AttackGenerativeModel(profile, [template])
    sample = list(model.sample(3, random.Random(1)))
    times = [datetime.fromisoformat(row["timestamp"]) for row in sample]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    if any(gap < 61.0 for gap in gaps):
        print(f"FAIL: expected synthetic attack reports to be spaced, got gaps {gaps}")
        return 1

    print("PASS: attack report sequence and payload ctlNum are rewritten separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
