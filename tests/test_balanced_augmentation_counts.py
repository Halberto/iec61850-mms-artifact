#!/usr/bin/env python3
"""Regression test for exact class budgets in balanced MMS augmentation.

Run: python tests/test_balanced_augmentation_counts.py
"""
import random
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "augmentation"))

from mms_dataset_augmentor import (  # noqa: E402
    AttackProfile,
    EventContextProfile,
    NormalProfile,
    generate_attacks,
    generate_normal,
    plan_isolated_attack_times,
    plan_proportional_attack_times,
    proportional_attack_count,
    replay_normal_baseline,
)


FIELDNAMES = [
    "frame_number", "timestamp", "tag", "direction", "service",
    "src_ip", "src_port", "dst_ip", "dst_port", "stream_id", "invoke_id",
    "variable_list_name", "variables", "control_object", "control_action",
    "control_value", "ctl_num", "origin_identifier", "origin_category",
    "octet_identities", "access_result",
]


def row(**overrides):
    base = {
        "frame_number": "1",
        "timestamp": "2026-02-14T01:00:00.000000",
        "tag": "normal",
        "direction": "RESPONSE",
        "service": "UNCONFIRMED",
        "src_ip": "10.0.19.49",
        "src_port": "102",
        "dst_ip": "10.0.19.39",
        "dst_port": "56638",
        "stream_id": "10.0.19.39:56638-10.0.19.49:102",
        "invoke_id": "",
        "variable_list_name": "RPT",
        "variables": "",
        "control_object": "",
        "control_action": "",
        "control_value": "",
        "ctl_num": "",
        "origin_identifier": "",
        "origin_category": "",
        "octet_identities": "",
        "access_result": "[]",
    }
    base.update(overrides)
    return base


def main() -> int:
    rng = random.Random(7)
    start = datetime(2026, 2, 14, 1, 0, 0)
    end = start + timedelta(hours=2)

    attack_profiles = {
        "E03A103_CTRL/LLN0$RP$A_URCB_10": AttackProfile(
            urcb_ref="E03A103_CTRL/LLN0$RP$A_URCB_10",
            burst_sizes=[1, 3, 5],
            intra_deltas=[0.1, 0.2],
            combo_counts={("E03A103_CTRL/Q1CSWI1$ST$Pos",): 2},
            max_seq=10,
        ),
        "E01A103_CTRL/LLN0$RP$A_URCB_10": AttackProfile(
            urcb_ref="E01A103_CTRL/LLN0$RP$A_URCB_10",
            burst_sizes=[2, 7],
            intra_deltas=[0.1, 0.3],
            combo_counts={("E01A103_CTRL/Q0CSWI1$ST$Pos",): 3},
            max_seq=20,
        ),
    }
    attacks = generate_attacks(attack_profiles, 37, start, start + timedelta(minutes=5), [1000], rng)
    if len(attacks) != 37:
        print(f"FAIL: expected 37 synthetic attacks, got {len(attacks)}")
        return 1
    if {r.get("tag") for r in attacks} != {"attack"}:
        print("FAIL: attack generator emitted a non-attack row")
        return 1
    attack_times = [
        datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
        for r in sorted(attacks, key=lambda row: row["timestamp"])
    ]
    close_pairs = [
        (a, b)
        for a, b in zip(attack_times, attack_times[1:])
        if (b - a).total_seconds() <= 60.0
    ]
    if close_pairs:
        print(f"FAIL: attack generator emitted burst-like close pairs: {close_pairs[:3]}")
        return 1
    if (attack_times[-1] - start).total_seconds() < 37 * 61.0:
        print("FAIL: attack generator compressed attacks instead of extending the schedule")
        return 1

    sbow = row(
        direction="REQUEST", service="WRITE", src_ip="10.0.19.39",
        src_port="50000", dst_ip="10.0.19.49", dst_port="102",
        control_action="SBOw", control_object="E03A103_CTRL/Q0CSWI1$CO$Pos",
        ctl_num="1", variable_list_name="",
    )
    oper = row(
        direction="REQUEST", service="WRITE", src_ip="10.0.19.39",
        src_port="50000", dst_ip="10.0.19.49", dst_port="102",
        control_action="Oper", control_object="E03A103_CTRL/Q0CSWI1$CO$Pos",
        ctl_num="1", variable_list_name="",
    )
    gnl = row(
        direction="REQUEST", service="GET_NAME_LIST", src_ip="10.0.19.39",
        src_port="50000", dst_ip="10.0.19.49", dst_port="102",
        variable_list_name="", stream_id="10.0.19.39:50000-10.0.19.49:102",
        invoke_id="10",
    )
    report_a = row(stream_id="stream-a")
    report_b = row(stream_id="stream-b", src_ip="10.0.19.47", dst_port="56633")
    other = row(service="IDENTIFY", variable_list_name="")
    normal_profile = NormalProfile(
        control_templates=[(sbow, oper)],
        sbow_oper_deltas=[3.5],
        req_resp_rtts=[0.02],
        gnl_templates=[gnl],
        report_templates=[report_a, report_b],
        report_stream_counts={"stream-a": 3, "stream-b": 1},
        other_templates=[other],
        n_control_rows=4,
        n_gnl_rows=8,
        n_report_rows=12,
        n_other_rows=1,
    )
    normals = generate_normal(normal_profile, 53, start, end, [2000], [3000], rng, FIELDNAMES, [10])
    if len(normals) != 53:
        print(f"FAIL: expected 53 synthetic normals, got {len(normals)}")
        return 1
    if {r.get("tag") for r in normals} != {"normal"}:
        print("FAIL: normal generator emitted a non-normal row")
        return 1

    by_service = Counter(r["service"] for r in normals)
    if by_service["WRITE"] % 4:
        print(f"FAIL: WRITE rows should preserve 4-row control groups, got {by_service['WRITE']}")
        return 1
    if by_service["GET_NAME_LIST"] % 2:
        print(f"FAIL: GET_NAME_LIST rows should preserve request/response pairs, got {by_service['GET_NAME_LIST']}")
        return 1

    write_requests = [
        int(r["ctl_num"])
        for r in sorted(normals, key=lambda r: r["timestamp"])
        if r["service"] == "WRITE" and r["direction"] == "REQUEST"
    ]
    expected = []
    for ctl in range(10, 10 + (len(write_requests) // 2)):
        expected.extend([ctl, ctl])
    if write_requests != expected:
        print(f"FAIL: WRITE request ctl_num should be sequential pairs, got {write_requests}")
        return 1

    context = EventContextProfile(
        attack_per_normal_row=0.02,
        oper_resp_to_report_deltas=[0.05],
    )
    if proportional_attack_count(context, 500) != 10:
        print("FAIL: proportional attack count did not follow baseline attack-per-normal rate")
        return 1
    planned = plan_proportional_attack_times(
        normals,
        10,
        context,
        start,
        end,
        rng,
        min_attack_gap=61.0,
    )
    gaps = [(b - a).total_seconds() for a, b in zip(planned, planned[1:])]
    if len(planned) != 10 or any(gap < 61.0 for gap in gaps):
        print(f"FAIL: proportional attack planner produced invalid times: {planned}")
        return 1

    isolated = plan_isolated_attack_times(
        1000,
        start,
        start + timedelta(days=1),
        rng,
        min_attack_gap=61.0,
    )
    isolated_gaps = [(b - a).total_seconds() for a, b in zip(isolated, isolated[1:])]
    if len(isolated) != 1000 or any(gap < 61.0 for gap in isolated_gaps):
        print("FAIL: exact-target attack scheduler compressed or dropped attacks")
        return 1
    if len({round(gap, 3) for gap in isolated_gaps[:50]}) <= 3:
        print(f"FAIL: exact-target attack scheduler produced a mechanical cadence: {isolated_gaps[:10]}")
        return 1

    replay_source = [
        row(
            frame_number="1", timestamp=start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            direction="REQUEST", service="WRITE", invoke_id="1",
            control_action="SBOw", control_object="E03A103_CTRL/Q0CSWI1$CO$Pos",
            ctl_num="5",
        ),
        row(
            frame_number="2", timestamp=(start + timedelta(milliseconds=20)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            direction="RESPONSE", service="WRITE", invoke_id="1",
            access_result="",
        ),
        row(
            frame_number="3", timestamp=(start + timedelta(seconds=3)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            direction="REQUEST", service="WRITE", invoke_id="2",
            control_action="Oper", control_object="E03A103_CTRL/Q0CSWI1$CO$Pos",
            ctl_num="5",
        ),
        row(
            frame_number="4", timestamp=(start + timedelta(seconds=3, milliseconds=20)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            direction="RESPONSE", service="WRITE", invoke_id="2",
            access_result=json.dumps([{"type": "unsigned", "value": 5}]),
        ),
    ]
    replayed = replay_normal_baseline(
        replay_source, 4, start, start + timedelta(minutes=10),
        [9000], [8000], [200], FIELDNAMES,
    )
    replay_invokes = [r["invoke_id"] for r in replayed]
    if replay_invokes != ["8000", "8000", "8001", "8001"]:
        print(f"FAIL: replay did not preserve request/response invoke pairs: {replay_invokes}")
        return 1
    replay_ctls = [
        int(r["ctl_num"]) for r in replayed
        if r["service"] == "WRITE" and r["direction"] == "REQUEST"
    ]
    if replay_ctls != [200, 200]:
        print(f"FAIL: replay did not rewrite control numbers as sequential pairs: {replay_ctls}")
        return 1
    if '"value":200' not in replayed[-1]["access_result"]:
        print(f"FAIL: replay did not rewrite nested control payload: {replayed[-1]['access_result']}")
        return 1

    print("PASS: balanced augmentation generators honor exact counts and protocol groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
