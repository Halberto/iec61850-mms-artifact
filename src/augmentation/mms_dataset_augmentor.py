"""
MMS IEC 61850 Dataset Augmentation Framework  (v2 — balanced synthesis)
========================================================================

Grows the full labelled dataset (mms_capture_attack_tags.csv) by generating
synthetic rows for BOTH normal traffic and attack traffic. By default, normal
traffic is scaled and attack reports are injected proportionally to the
baseline attack-per-normal rate, then placed near synthetic control/report
contexts so the augmented timeline keeps the original protocol mechanics.

An explicit ratio mode remains available for experiments, but proportional
mode is the realistic default.

Normal synthesis preserves the three real traffic types in their observed
proportions:
  Type A — GET_NAME_LIST request/response pairs (periodic IED enumeration)
  Type B — Control sequences: SBOw_req -> SBOw_resp -> Oper_req -> Oper_resp
            -> UNCONFIRMED report with originator (legitimate earth-switch ops)
  Type C — Autonomous UNCONFIRMED reports (periodic state, no originator)

Attack synthesis uses isolated report generation: it preserves the real URCB,
object-reference, and sequence-number patterns without creating artificial
IEDEXPLORER bursts.

The capture time window is extended proportionally to the scale factor so that
traffic density stays realistic rather than cramming everything into 51 hours.

Feature Taxonomy (10 features driving both normal and attack synthesis):
  F1  src_ip              Network: source IED or SCADA IP
  F2  dst_ip              Network: destination IP
  F3  stream_id           Network: TCP association identifier
  F4  service             Protocol: WRITE / GET_NAME_LIST / UNCONFIRMED
  F5  direction           Protocol: REQUEST (SCADA->IED) / RESPONSE (IED->SCADA)
  F6  variable_list_name  Protocol: RPT for reports
  F7  octet_identities    Authorization: IEDEXPLORER = attack identity
  F8  urcb_ref            IEC 61850: Report Control Block (from access_result[0])
  F9  report_seq_num      IEC 61850: monotonic sequence per URCB
  F10 data_refs           IEC 61850: control object data references

Usage:
    python src/augmentation/mms_dataset_augmentor.py \\
        --input  data/raw/mms_capture_attack_tags.csv \\
        --output data/raw/mms_capture_augmented.csv.gz \\
        --scale  4 \\
        --seed   42
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(10 ** 8)

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"
TS_FMT_SHORT = "%Y-%m-%dT%H:%M:%S"

ATTACK_TAG = "attack"
NORMAL_TAG = "normal"
MIN_SYNTHETIC_ATTACK_GAP_SECONDS = 61.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        if "w" in mode:
            return gzip.open(path, mode, encoding="utf-8", newline="", compresslevel=1)
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def _ts(s: str) -> datetime:
    try:
        return datetime.strptime(s, TS_FMT)
    except ValueError:
        return datetime.strptime(s, TS_FMT_SHORT)


def _ts_str(dt: datetime) -> str:
    return dt.strftime(TS_FMT)


def _weighted_choice(rng: random.Random, counts: dict):
    population = [item for item, w in counts.items() for _ in range(w)]
    return rng.choice(population)


def _allocate_counts(n_total: int, weights: dict) -> dict:
    """Allocate an exact integer total according to empirical weights."""
    if n_total <= 0:
        return {key: 0 for key in weights}

    active = {key: int(value) for key, value in weights.items() if int(value) > 0}
    if not active:
        return {key: 0 for key in weights}

    weight_total = sum(active.values())
    raw = {key: n_total * value / weight_total for key, value in active.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remainder = n_total - sum(allocated.values())
    order = sorted(
        active,
        key=lambda key: (raw[key] - allocated[key], active[key]),
        reverse=True,
    )
    for key in order[:remainder]:
        allocated[key] += 1

    return {key: int(allocated.get(key, 0)) for key in weights}


def _clamp_ts(value: datetime, start: datetime, end: datetime, margin_seconds: float = 0.0) -> datetime:
    latest = end - timedelta(seconds=margin_seconds)
    if latest < start:
        latest = end
    return max(start, min(latest, value))


def _median(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _mean(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _blank_from_template(template: dict, fieldnames: List[str]) -> dict:
    row = {k: "" for k in fieldnames}
    row.update({k: v for k, v in template.items() if k in fieldnames})
    return row


def _rewrite_json_field(value: str, ctl_value_map: Dict[str, str], new_ts: datetime) -> str:
    if not value:
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value

    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if not isinstance(node, dict):
            return node

        updated = dict(node)
        node_type = updated.get("type")
        if node_type == "unsigned":
            old_value = str(updated.get("value", ""))
            if old_value in ctl_value_map:
                new_value = ctl_value_map[old_value]
                updated["value"] = int(new_value) if new_value.isdigit() else new_value
        elif node_type == "utc-time":
            updated["value"] = new_ts.strftime(TS_FMT_SHORT)

        if "value" in updated:
            updated["value"] = rewrite(updated["value"])
        return updated

    return json.dumps(rewrite(parsed), separators=(",", ":"))


def _maybe_rewrite_json_field(value: str, ctl_value_map: Dict[str, str], new_ts: datetime) -> str:
    if not value:
        return value
    if not ctl_value_map and "utc-time" not in value:
        return value
    return _rewrite_json_field(value, ctl_value_map, new_ts)


def _normal_without_controls(prof: "NormalProfile") -> "NormalProfile":
    return NormalProfile(
        gnl_templates=prof.gnl_templates,
        gnl_inter_arrivals=prof.gnl_inter_arrivals,
        report_templates=prof.report_templates,
        report_inter_arrivals=prof.report_inter_arrivals,
        report_stream_counts=prof.report_stream_counts,
        other_templates=prof.other_templates,
        other_inter_arrivals=prof.other_inter_arrivals,
        n_gnl_rows=prof.n_gnl_rows,
        n_report_rows=prof.n_report_rows,
        n_other_rows=prof.n_other_rows,
    )


def _is_normal_request_response(row: dict) -> bool:
    return row.get("tag") == NORMAL_TAG and row.get("service") != "UNCONFIRMED"


def _is_normal_report(row: dict) -> bool:
    return row.get("tag") == NORMAL_TAG and row.get("service") == "UNCONFIRMED"


def _normal_subtype_counts(rows: List[dict]) -> Tuple[int, int]:
    request_response = sum(1 for row in rows if _is_normal_request_response(row))
    reports = sum(1 for row in rows if _is_normal_report(row))
    return request_response, reports


# ── Attack profile & synthesis (F7-F10) ──────────────────────────────────────

URCB_ENVELOPE: Dict[str, dict] = {
    "E03A103_CTRL/LLN0$RP$A_URCB_10": {
        "src_ip": "10.0.19.49", "src_port": "102",
        "dst_ip": "10.0.19.39", "dst_port": "56638",
        "stream_id": "10.0.19.39:56638-10.0.19.49:102",
        "entry_id": "E03A103_CTRL/LLN0$08002744E921ST0",
    },
    "E01A103_CTRL/LLN0$RP$A_URCB_10": {
        "src_ip": "10.0.19.47", "src_port": "102",
        "dst_ip": "10.0.19.39", "dst_port": "56633",
        "stream_id": "10.0.19.39:56633-10.0.19.47:102",
        "entry_id": "E01A103_CTRL/LLN0$08002744E921ST0",
    },
}


@dataclass
class AttackProfile:
    urcb_ref: str
    burst_sizes: List[int] = field(default_factory=list)
    intra_deltas: List[float] = field(default_factory=list)
    combo_counts: Dict[Tuple, int] = field(default_factory=dict)
    max_seq: int = 0


def profile_attacks(rows: List[dict]) -> Dict[str, AttackProfile]:
    profiles: Dict[str, AttackProfile] = {u: AttackProfile(u) for u in URCB_ENVELOPE}
    atk_rows = []
    for row in rows:
        if row.get("tag") != ATTACK_TAG:
            continue
        ar = json.loads(row["access_result"])
        urcb = ar[0]["value"]
        seq = next(e["value"] for e in ar if e["type"] == "unsigned")
        combo = tuple(sorted(e["value"] for e in ar[2:] if e["type"] == "visible-string"))
        atk_rows.append((row["timestamp"], urcb, seq, combo))

    atk_rows.sort()

    for urcb, prof in profiles.items():
        stream = [(t, s, c) for t, u, s, c in atk_rows if u == urcb]
        if not stream:
            continue
        prof.max_seq = max(s for _, s, _ in stream)

        bursts: List[List[tuple]] = [[stream[0]]]
        for prev, cur in zip(stream, stream[1:]):
            gap = (_ts(cur[0]) - _ts(prev[0])).total_seconds()
            if gap > 60:
                bursts.append([cur])
            else:
                bursts[-1].append(cur)

        for burst in bursts:
            prof.burst_sizes.append(len(burst))
            for a, b in zip(burst, burst[1:]):
                prof.intra_deltas.append((_ts(b[0]) - _ts(a[0])).total_seconds())
            for _, _, c in burst:
                prof.combo_counts[c] = prof.combo_counts.get(c, 0) + 1

    return profiles


def _make_attack_row(
    urcb_ref: str, seq: int, combo: Tuple, ts: datetime, frame: int
) -> dict:
    env = URCB_ENVELOPE[urcb_ref]
    entry_id = env["entry_id"]
    data_refs = [r for r in combo if r != entry_id]
    ar = [{"type": "visible-string", "value": urcb_ref},
          {"type": "unsigned", "value": seq},
          {"type": "visible-string", "value": entry_id}]
    for ref in data_refs:
        ar.append({"type": "visible-string", "value": ref})
    return {
        "frame_number": str(frame), "timestamp": _ts_str(ts),
        "tag": ATTACK_TAG, "direction": "RESPONSE", "service": "UNCONFIRMED",
        "src_ip": env["src_ip"], "src_port": env["src_port"],
        "dst_ip": env["dst_ip"], "dst_port": env["dst_port"],
        "stream_id": env["stream_id"], "invoke_id": "",
        "variable_list_name": "RPT", "variables": "", "control_object": "",
        "control_action": "", "control_value": "", "ctl_num": "",
        "origin_identifier": "", "origin_category": "",
        "octet_identities": "IEDEXPLORER",
        "access_result": json.dumps(ar, separators=(",", ":")),
    }


def generate_attacks(
    profiles: Dict[str, AttackProfile],
    n_total: int,
    window_start: datetime,
    window_end: datetime,
    frame_counter: list,
    rng: random.Random,
    min_attack_gap: float = MIN_SYNTHETIC_ATTACK_GAP_SECONDS,
    scheduled_times: Optional[List[datetime]] = None,
) -> List[dict]:
    if scheduled_times is not None:
        n_total = len(scheduled_times)
    if n_total <= 0:
        return []

    weights = {u: sum(p.burst_sizes) for u, p in profiles.items() if p.burst_sizes}
    targets = _allocate_counts(n_total, weights)
    seq_counters = {u: p.max_seq + 1 for u, p in profiles.items()}
    result: List[dict] = []
    schedule: List[str] = []

    for urcb_ref, synth_n in targets.items():
        if synth_n <= 0:
            continue
        schedule.extend([urcb_ref] * synth_n)

    rng.shuffle(schedule)
    if scheduled_times is None:
        window_secs = max((window_end - window_start).total_seconds(), 0.001)
        slot = max(float(min_attack_gap), window_secs / (len(schedule) + 1))
        attack_times = [window_start + timedelta(seconds=slot * (i + 1)) for i in range(len(schedule))]
    else:
        attack_times = sorted(scheduled_times)

    for urcb_ref, ts in zip(schedule, attack_times):
        prof = profiles[urcb_ref]
        combo = _weighted_choice(rng, prof.combo_counts)
        row = _make_attack_row(urcb_ref, seq_counters[urcb_ref], combo, ts, frame_counter[0])
        result.append(row)
        frame_counter[0] += 1
        seq_counters[urcb_ref] += 1

    return result


# ── Normal traffic profile & synthesis (F1-F6, F8-F10) ───────────────────────

@dataclass
class NormalProfile:
    # Control sequences: SBOw + Oper pairs
    control_templates: List[Tuple[dict, dict]] = field(default_factory=list)  # (sbow, oper)
    sbow_oper_deltas: List[float] = field(default_factory=list)
    req_resp_rtts: List[float] = field(default_factory=list)
    # GET_NAME_LIST templates (request rows)
    gnl_templates: List[dict] = field(default_factory=list)
    gnl_inter_arrivals: List[float] = field(default_factory=list)
    # Autonomous report templates (no originator in access_result)
    report_templates: List[dict] = field(default_factory=list)
    report_inter_arrivals: Dict[str, List[float]] = field(default_factory=dict)
    report_stream_counts: Dict[str, int] = field(default_factory=dict)
    # Rare normal MMS service rows outside the main three traffic types.
    other_templates: List[dict] = field(default_factory=list)
    other_inter_arrivals: List[float] = field(default_factory=list)
    # Proportions in original data
    n_control_rows: int = 0
    n_gnl_rows: int = 0
    n_report_rows: int = 0
    n_other_rows: int = 0


@dataclass
class EventContextProfile:
    normal_control_events: int = 0
    attack_reports: int = 0
    attack_per_normal_row: float = 0.0
    attack_per_control_event: float = 0.0
    oper_resp_to_report_deltas: List[float] = field(default_factory=list)
    normal_report_rows_before_60s: List[int] = field(default_factory=list)
    normal_report_rows_after_60s: List[int] = field(default_factory=list)
    attack_rows_before_60s: List[int] = field(default_factory=list)
    attack_rows_after_60s: List[int] = field(default_factory=list)


def _ied_from_control_object(value: str) -> str:
    return (value or "").split("/", 1)[0]


def _ied_from_report(row: dict) -> str:
    access_raw = row.get("access_result") or ""
    for marker in ("E03A103_CTRL", "E01A103_CTRL", "E02A103_CTRL", "E03A101_CTRL", "E01A101_CTRL"):
        if marker in access_raw:
            return marker
    src = row.get("src_ip") or ""
    if src == "10.0.19.49":
        return "E03A103_CTRL"
    if src == "10.0.19.47":
        return "E01A103_CTRL"
    return ""


def _count_rows_in_window(rows: List[dict], center: dict, before: bool, seconds: float) -> int:
    center_ts = center["_dt"]
    center_pos = center["_pos"]
    count = 0
    if before:
        pos = center_pos - 1
        while pos >= 0 and (center_ts - rows[pos]["_dt"]).total_seconds() <= seconds:
            count += 1
            pos -= 1
    else:
        pos = center_pos + 1
        while pos < len(rows) and (rows[pos]["_dt"] - center_ts).total_seconds() <= seconds:
            count += 1
            pos += 1
    return count


def profile_event_context(rows: List[dict]) -> EventContextProfile:
    prof = EventContextProfile()
    timed_rows = [dict(r) for r in rows if r.get("timestamp")]
    timed_rows.sort(key=lambda r: (_ts(r["timestamp"]), int(r.get("frame_number") or 0)))
    for pos, row in enumerate(timed_rows):
        row["_pos"] = pos
        row["_dt"] = _ts(row["timestamp"])

    normal_rows = [r for r in timed_rows if r.get("tag") == NORMAL_TAG]
    attack_rows = [r for r in timed_rows if r.get("tag") == ATTACK_TAG]
    prof.attack_reports = len(attack_rows)
    prof.attack_per_normal_row = len(attack_rows) / len(normal_rows) if normal_rows else 0.0

    by_invoke: Dict[str, List[dict]] = defaultdict(list)
    opers_by_key: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    sbows: List[dict] = []
    reports_by_ied: Dict[str, List[dict]] = defaultdict(list)

    for row in timed_rows:
        if row.get("invoke_id"):
            by_invoke[row["invoke_id"]].append(row)
        if row.get("tag") == NORMAL_TAG and row.get("direction") == "REQUEST" and row.get("service") == "WRITE":
            key = (row.get("ctl_num", ""), row.get("control_object", ""))
            if row.get("control_action") == "SBOw":
                sbows.append(row)
            elif row.get("control_action") == "Oper":
                opers_by_key[key].append(row)
        if row.get("tag") == NORMAL_TAG and row.get("service") == "UNCONFIRMED":
            ied = _ied_from_report(row)
            if ied:
                reports_by_ied[ied].append(row)

    for s in sbows:
        key = (s.get("ctl_num", ""), s.get("control_object", ""))
        oper_candidates = [r for r in opers_by_key.get(key, []) if r["_pos"] > s["_pos"]]
        if not oper_candidates:
            continue
        oper = min(oper_candidates, key=lambda r: r["_pos"])
        oper_resp = min(
            [
                r for r in by_invoke.get(oper.get("invoke_id", ""), [])
                if r.get("direction") == "RESPONSE" and r["_pos"] > oper["_pos"]
            ],
            key=lambda r: r["_pos"],
            default=None,
        )
        anchor = oper_resp or oper
        ied = _ied_from_control_object(s.get("control_object", ""))
        report = min(
            [r for r in reports_by_ied.get(ied, []) if r["_pos"] > anchor["_pos"]],
            key=lambda r: r["_pos"],
            default=None,
        )
        if not report:
            continue
        prof.normal_control_events += 1
        prof.oper_resp_to_report_deltas.append((report["_dt"] - anchor["_dt"]).total_seconds())
        prof.normal_report_rows_before_60s.append(_count_rows_in_window(timed_rows, report, True, 60.0))
        prof.normal_report_rows_after_60s.append(_count_rows_in_window(timed_rows, report, False, 60.0))

    prof.attack_per_control_event = (
        prof.attack_reports / prof.normal_control_events if prof.normal_control_events else 0.0
    )
    for attack in attack_rows:
        prof.attack_rows_before_60s.append(_count_rows_in_window(timed_rows, attack, True, 60.0))
        prof.attack_rows_after_60s.append(_count_rows_in_window(timed_rows, attack, False, 60.0))
    return prof


def profile_normal(rows: List[dict]) -> NormalProfile:
    prof = NormalProfile()
    normal = [r for r in rows if r.get("tag") == NORMAL_TAG]

    write_req = {r["invoke_id"]: r for r in normal
                 if r["direction"] == "REQUEST" and r["service"] == "WRITE"}
    write_resp = {r["invoke_id"]: r for r in normal
                  if r["direction"] == "RESPONSE" and r["service"] == "WRITE"}

    # RTT: req -> resp delta
    for inv in set(write_req) & set(write_resp):
        dt = (_ts(write_resp[inv]["timestamp"]) - _ts(write_req[inv]["timestamp"])).total_seconds()
        if 0 < dt < 10:
            prof.req_resp_rtts.append(dt)

    # SBOw + Oper pair templates
    sbows = [r for r in normal if r.get("control_action") == "SBOw"]
    opers = {(r["ctl_num"], r["control_object"]): r for r in normal
             if r.get("control_action") == "Oper"}
    for s in sbows:
        key = (s["ctl_num"], s["control_object"])
        if key in opers:
            o = opers[key]
            prof.control_templates.append((s, o))
            dt = (_ts(o["timestamp"]) - _ts(s["timestamp"])).total_seconds()
            if 0 < dt < 60:
                prof.sbow_oper_deltas.append(dt)

    prof.n_control_rows = len([r for r in normal
                                if r["service"] == "WRITE" or r.get("control_action") in ("SBOw", "Oper")])

    # GET_NAME_LIST templates
    gnl_sorted = sorted(
        [r for r in normal if r["service"] == "GET_NAME_LIST" and r["direction"] == "REQUEST"],
        key=lambda r: r["timestamp"]
    )
    prof.gnl_templates = gnl_sorted
    for a, b in zip(gnl_sorted, gnl_sorted[1:]):
        dt = (_ts(b["timestamp"]) - _ts(a["timestamp"])).total_seconds()
        if 0 < dt < 30:
            prof.gnl_inter_arrivals.append(dt)
    prof.n_gnl_rows = len([r for r in normal if r["service"] == "GET_NAME_LIST"])

    # Autonomous UNCONFIRMED report templates (no originator structure)
    all_reports = [r for r in normal if r["service"] == "UNCONFIRMED"]
    prof.report_templates = all_reports
    # Per-stream inter-arrivals
    by_stream: Dict[str, List[datetime]] = defaultdict(list)
    for r in sorted(all_reports, key=lambda x: x["timestamp"]):
        by_stream[r["stream_id"]].append(_ts(r["timestamp"]))
    for sid, times in by_stream.items():
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        gaps = [g for g in gaps if 0 < g < 60]
        if gaps:
            prof.report_inter_arrivals[sid] = gaps
    prof.report_stream_counts = dict(Counter(r["stream_id"] for r in all_reports))
    prof.n_report_rows = len(all_reports)

    prof.other_templates = [
        r for r in normal
        if r["service"] not in ("WRITE", "GET_NAME_LIST", "UNCONFIRMED")
        and r.get("control_action") not in ("SBOw", "Oper")
    ]
    other_sorted = sorted(prof.other_templates, key=lambda r: r["timestamp"])
    for a, b in zip(other_sorted, other_sorted[1:]):
        dt = (_ts(b["timestamp"]) - _ts(a["timestamp"])).total_seconds()
        if 0 < dt < 60:
            prof.other_inter_arrivals.append(dt)
    prof.n_other_rows = len(prof.other_templates)

    return prof


def replay_normal_baseline(
    rows: List[dict],
    n_total: int,
    orig_start: datetime,
    orig_end: datetime,
    frame_counter: list,
    invoke_counter: list,
    control_counter: list,
    fieldnames: List[str],
) -> List[dict]:
    """Replay real normal rows in timestamp order to preserve local packet mechanics."""
    if n_total <= 0:
        return []

    baseline = sorted(
        [r for r in rows if r.get("tag") == NORMAL_TAG and r.get("timestamp")],
        key=lambda r: (_ts(r["timestamp"]), int(r.get("frame_number") or 0)),
    )
    if not baseline:
        return []

    duration = max((orig_end - orig_start).total_seconds(), 0.001)
    result: List[dict] = []
    copy_index = 1

    while len(result) < n_total:
        invoke_map: Dict[str, str] = {}
        ctl_event_map: Dict[Tuple[str, str], str] = {}
        ctl_value_map: Dict[str, str] = {}

        for template in baseline:
            if len(result) >= n_total:
                break

            old_ts = _ts(template["timestamp"])
            new_ts = old_ts + timedelta(seconds=duration * copy_index)
            row = _blank_from_template(template, fieldnames)
            row["frame_number"] = str(frame_counter[0])
            frame_counter[0] += 1
            row["timestamp"] = _ts_str(new_ts)
            row["tag"] = NORMAL_TAG

            old_invoke = row.get("invoke_id", "")
            if old_invoke:
                if old_invoke not in invoke_map:
                    invoke_map[old_invoke] = str(invoke_counter[0])
                    invoke_counter[0] += 1
                row["invoke_id"] = invoke_map[old_invoke]

            old_ctl = row.get("ctl_num", "")
            if old_ctl and str(old_ctl).isdigit():
                ctl_key = (str(old_ctl), row.get("control_object", ""))
                if ctl_key not in ctl_event_map:
                    ctl_event_map[ctl_key] = str(control_counter[0])
                    control_counter[0] += 1
                    ctl_value_map[str(old_ctl)] = ctl_event_map[ctl_key]
                row["ctl_num"] = ctl_event_map[ctl_key]

            if row.get("variables"):
                row["variables"] = _maybe_rewrite_json_field(row["variables"], ctl_value_map, new_ts)
            if row.get("access_result"):
                row["access_result"] = _maybe_rewrite_json_field(row["access_result"], ctl_value_map, new_ts)

            result.append(row)

        copy_index += 1

    return result


def replay_normal_baseline_by_targets(
    rows: List[dict],
    n_request_response: int,
    n_reports: int,
    orig_start: datetime,
    orig_end: datetime,
    window_start: datetime,
    window_end: datetime,
    frame_counter: list,
    invoke_counter: list,
    control_counter: list,
    fieldnames: List[str],
) -> List[dict]:
    """Replay normal baseline rows while preserving request/response and report budgets."""
    if n_request_response <= 0 and n_reports <= 0:
        return []

    baseline = sorted(
        [r for r in rows if r.get("tag") == NORMAL_TAG and r.get("timestamp")],
        key=lambda r: (_ts(r["timestamp"]), int(r.get("frame_number") or 0)),
    )
    if not baseline:
        return []

    duration = max((orig_end - orig_start).total_seconds(), 0.001)
    result: List[dict] = []
    remaining_rr = n_request_response
    remaining_reports = n_reports
    copy_index = 1

    while remaining_rr > 0 or remaining_reports > 0:
        emitted_this_pass = False
        invoke_map: Dict[str, str] = {}
        ctl_event_map: Dict[Tuple[str, str], str] = {}
        ctl_value_map: Dict[str, str] = {}

        for template in baseline:
            is_rr = _is_normal_request_response(template)
            is_report = _is_normal_report(template)
            if is_rr:
                if remaining_rr <= 0:
                    continue
                remaining_rr -= 1
            elif is_report:
                if remaining_reports <= 0:
                    continue
                remaining_reports -= 1
            else:
                continue

            old_ts = _ts(template["timestamp"])
            new_ts = old_ts + timedelta(seconds=duration * copy_index)
            row = _blank_from_template(template, fieldnames)
            row["frame_number"] = str(frame_counter[0])
            frame_counter[0] += 1
            row["timestamp"] = _ts_str(new_ts)
            row["tag"] = NORMAL_TAG

            old_invoke = row.get("invoke_id", "")
            if old_invoke:
                if old_invoke not in invoke_map:
                    invoke_map[old_invoke] = str(invoke_counter[0])
                    invoke_counter[0] += 1
                row["invoke_id"] = invoke_map[old_invoke]

            old_ctl = row.get("ctl_num", "")
            if old_ctl and str(old_ctl).isdigit():
                ctl_key = (str(old_ctl), row.get("control_object", ""))
                if ctl_key not in ctl_event_map:
                    ctl_event_map[ctl_key] = str(control_counter[0])
                    control_counter[0] += 1
                    ctl_value_map[str(old_ctl)] = ctl_event_map[ctl_key]
                row["ctl_num"] = ctl_event_map[ctl_key]

            if row.get("variables"):
                row["variables"] = _maybe_rewrite_json_field(row["variables"], ctl_value_map, new_ts)
            if row.get("access_result"):
                row["access_result"] = _maybe_rewrite_json_field(row["access_result"], ctl_value_map, new_ts)

            result.append(row)
            emitted_this_pass = True

            if remaining_rr <= 0 and remaining_reports <= 0:
                break

        if not emitted_this_pass:
            break
        copy_index += 1

    if len(result) > 1 and window_end > window_start:
        original_times = [_ts(row["timestamp"]) for row in result]
        first_ts = original_times[0]
        last_ts = original_times[-1]
        span = max((last_ts - first_ts).total_seconds(), 0.001)
        target_span = max((window_end - window_start).total_seconds(), 0.001)
        for row, old_ts in zip(result, original_times):
            offset = (old_ts - first_ts).total_seconds() / span
            new_ts = window_start + timedelta(seconds=offset * target_span)
            row["timestamp"] = _ts_str(new_ts)
            if row.get("variables"):
                row["variables"] = _maybe_rewrite_json_field(row["variables"], {}, new_ts)
            if row.get("access_result"):
                row["access_result"] = _maybe_rewrite_json_field(row["access_result"], {}, new_ts)

    return result


def _generate_normal_legacy(
    prof: NormalProfile,
    n_total: int,
    window_start: datetime,
    window_end: datetime,
    frame_counter: list,
    invoke_counter: list,
    rng: random.Random,
    fieldnames: List[str],
) -> List[dict]:
    result: List[dict] = []

    # Proportion weights from real data
    total_real = prof.n_control_rows + prof.n_gnl_rows + prof.n_report_rows
    if total_real == 0:
        return result

    n_control = max(1, round(n_total * prof.n_control_rows / total_real))
    n_gnl = max(1, round(n_total * prof.n_gnl_rows / total_real))
    n_report = n_total - n_control - n_gnl

    window_secs = (window_end - window_start).total_seconds()

    def blank_row(template: dict) -> dict:
        row = {k: "" for k in fieldnames}
        row.update({k: v for k, v in template.items() if k in fieldnames})
        return row

    # ── Type B: Control sequences ─────────────────────────────────────────────
    # Each sequence = SBOw_req + SBOw_resp + Oper_req + Oper_resp (4 rows)
    n_sequences = max(1, n_control // 4)
    slot = window_secs / (n_sequences + 1)
    rtt = prof.req_resp_rtts
    sbow_oper = prof.sbow_oper_deltas

    for i in range(n_sequences):
        base = window_start + timedelta(seconds=slot * (i + 1))
        jitter = rng.uniform(-slot * 0.3, slot * 0.3)
        t0 = max(window_start, min(window_end - timedelta(seconds=30),
                                    base + timedelta(seconds=jitter)))

        sbow_tmpl, oper_tmpl = rng.choice(prof.control_templates)
        inv_sbow = str(invoke_counter[0]); invoke_counter[0] += 1
        inv_oper = str(invoke_counter[0]); invoke_counter[0] += 1

        rtt_val = rng.choice(rtt) if rtt else 0.05
        sbow_oper_val = rng.choice(sbow_oper) if sbow_oper else 3.5

        t_sbow_req = t0
        t_sbow_resp = t0 + timedelta(seconds=rtt_val)
        t_oper_req = t_sbow_resp + timedelta(seconds=sbow_oper_val)
        t_oper_resp = t_oper_req + timedelta(seconds=rtt_val)

        for tmpl, t, inv, direction, service in [
            (sbow_tmpl, t_sbow_req,  inv_sbow, "REQUEST",  "WRITE"),
            (sbow_tmpl, t_sbow_resp, inv_sbow, "RESPONSE", "WRITE"),
            (oper_tmpl, t_oper_req,  inv_oper, "REQUEST",  "WRITE"),
            (oper_tmpl, t_oper_resp, inv_oper, "RESPONSE", "WRITE"),
        ]:
            row = blank_row(tmpl)
            row["frame_number"] = str(frame_counter[0]); frame_counter[0] += 1
            row["timestamp"] = _ts_str(t)
            row["tag"] = NORMAL_TAG
            row["direction"] = direction
            row["service"] = service
            row["invoke_id"] = inv
            if direction == "RESPONSE":
                # Response: IED -> SCADA, flip src/dst
                row["src_ip"] = tmpl["dst_ip"]
                row["src_port"] = tmpl["dst_port"]
                row["dst_ip"] = tmpl["src_ip"]
                row["dst_port"] = tmpl["src_port"]
                row["control_object"] = ""
                row["control_action"] = ""
                row["control_value"] = ""
                row["ctl_num"] = ""
                row["origin_identifier"] = ""
                row["origin_category"] = ""
            result.append(row)

    # ── Type A: GET_NAME_LIST pairs ───────────────────────────────────────────
    n_gnl_pairs = n_gnl // 2
    slot_gnl = window_secs / (n_gnl_pairs + 1)
    for i in range(n_gnl_pairs):
        base = window_start + timedelta(seconds=slot_gnl * (i + 1))
        jitter = rng.uniform(-slot_gnl * 0.4, slot_gnl * 0.4)
        t0 = max(window_start, min(window_end - timedelta(seconds=5),
                                    base + timedelta(seconds=jitter)))
        tmpl = rng.choice(prof.gnl_templates)
        inv = str(invoke_counter[0]); invoke_counter[0] += 1
        rtt_val = rng.choice(rtt) if rtt else 0.05

        for direction, t_offset in [("REQUEST", 0.0), ("RESPONSE", rtt_val)]:
            row = blank_row(tmpl)
            row["frame_number"] = str(frame_counter[0]); frame_counter[0] += 1
            row["timestamp"] = _ts_str(t0 + timedelta(seconds=t_offset))
            row["tag"] = NORMAL_TAG
            row["direction"] = direction
            row["service"] = "GET_NAME_LIST"
            row["invoke_id"] = inv
            if direction == "RESPONSE":
                row["src_ip"] = tmpl["dst_ip"]
                row["src_port"] = tmpl["dst_port"]
                row["dst_ip"] = tmpl["src_ip"]
                row["dst_port"] = tmpl["src_port"]
            result.append(row)

    # ── Type C: Autonomous UNCONFIRMED reports ────────────────────────────────
    n_report_each = max(1, n_report // len(prof.report_inter_arrivals)) if prof.report_inter_arrivals else n_report
    for sid, gaps in prof.report_inter_arrivals.items():
        stream_reports = [r for r in prof.report_templates if r["stream_id"] == sid]
        if not stream_reports:
            continue
        slot_r = window_secs / (n_report_each + 1)
        for i in range(n_report_each):
            base = window_start + timedelta(seconds=slot_r * (i + 1))
            jitter = rng.uniform(-slot_r * 0.3, slot_r * 0.3)
            t = max(window_start, min(window_end - timedelta(milliseconds=1),
                                       base + timedelta(seconds=jitter)))
            tmpl = rng.choice(stream_reports)
            row = blank_row(tmpl)
            row["frame_number"] = str(frame_counter[0]); frame_counter[0] += 1
            row["timestamp"] = _ts_str(t)
            row["tag"] = NORMAL_TAG
            row["direction"] = "RESPONSE"
            row["service"] = "UNCONFIRMED"
            result.append(row)

    return result


def generate_normal(
    prof: NormalProfile,
    n_total: int,
    window_start: datetime,
    window_end: datetime,
    frame_counter: list,
    invoke_counter: list,
    rng: random.Random,
    fieldnames: List[str],
    control_counter: Optional[list] = None,
) -> List[dict]:
    if n_total <= 0:
        return []

    weights = {}
    if prof.control_templates:
        weights["control"] = prof.n_control_rows
    if prof.gnl_templates:
        weights["gnl"] = prof.n_gnl_rows
    if prof.report_templates:
        weights["report"] = prof.n_report_rows
    if prof.other_templates:
        weights["other"] = prof.n_other_rows
    if not weights:
        return []

    counts = _allocate_counts(n_total, weights)
    if "control" in counts:
        counts["control"] = (counts["control"] // 4) * 4
    if "gnl" in counts:
        counts["gnl"] = (counts["gnl"] // 2) * 2

    remainder = n_total - sum(counts.values())
    if remainder > 0 and "report" in counts:
        counts["report"] += remainder
        remainder = 0
    if remainder > 0 and "other" in counts:
        counts["other"] += remainder
        remainder = 0
    if remainder > 0 and "gnl" in counts:
        add = (remainder // 2) * 2
        counts["gnl"] += add
        remainder -= add
    if remainder > 0 and "control" in counts:
        add = (remainder // 4) * 4
        counts["control"] += add
        remainder -= add
    if remainder > 0:
        counts["fallback"] = remainder

    result: List[dict] = []
    window_secs = max((window_end - window_start).total_seconds(), 0.001)
    rtt = prof.req_resp_rtts
    sbow_oper = prof.sbow_oper_deltas
    if control_counter is None:
        max_ctl = max(
            (
                int(row.get("ctl_num", ""))
                for pair in prof.control_templates
                for row in pair
                if str(row.get("ctl_num", "")).isdigit()
            ),
            default=0,
        )
        control_counter = [max_ctl + 1]

    def blank_row(template: dict) -> dict:
        row = {k: "" for k in fieldnames}
        row.update({k: v for k, v in template.items() if k in fieldnames})
        return row

    def emit(template: dict, t: datetime, overrides: Optional[dict] = None) -> None:
        row = blank_row(template)
        row["frame_number"] = str(frame_counter[0])
        frame_counter[0] += 1
        row["timestamp"] = _ts_str(t)
        row["tag"] = NORMAL_TAG
        if overrides:
            row.update(overrides)
        result.append(row)

    n_sequences = counts.get("control", 0) // 4
    slot = window_secs / (n_sequences + 1) if n_sequences else window_secs
    for i in range(n_sequences):
        base = window_start + timedelta(seconds=slot * (i + 1))
        jitter = rng.uniform(-slot * 0.3, slot * 0.3)
        t0 = _clamp_ts(base + timedelta(seconds=jitter), window_start, window_end, 30)
        sbow_tmpl, oper_tmpl = rng.choice(prof.control_templates)
        inv_sbow = str(invoke_counter[0]); invoke_counter[0] += 1
        inv_oper = str(invoke_counter[0]); invoke_counter[0] += 1
        ctl_value = str(control_counter[0])
        control_counter[0] += 1
        rtt_val = rng.choice(rtt) if rtt else 0.05
        sbow_oper_val = rng.choice(sbow_oper) if sbow_oper else 3.5

        for tmpl, t, inv, direction in [
            (sbow_tmpl, t0, inv_sbow, "REQUEST"),
            (sbow_tmpl, t0 + timedelta(seconds=rtt_val), inv_sbow, "RESPONSE"),
            (oper_tmpl, t0 + timedelta(seconds=rtt_val + sbow_oper_val), inv_oper, "REQUEST"),
            (oper_tmpl, t0 + timedelta(seconds=(2 * rtt_val) + sbow_oper_val), inv_oper, "RESPONSE"),
        ]:
            overrides = {"direction": direction, "service": "WRITE", "invoke_id": inv}
            if direction == "REQUEST":
                overrides["ctl_num"] = ctl_value
            if direction == "RESPONSE":
                overrides.update({
                    "src_ip": tmpl["dst_ip"],
                    "src_port": tmpl["dst_port"],
                    "dst_ip": tmpl["src_ip"],
                    "dst_port": tmpl["src_port"],
                    "control_object": "",
                    "control_action": "",
                    "control_value": "",
                    "ctl_num": "",
                    "origin_identifier": "",
                    "origin_category": "",
                })
            emit(tmpl, _clamp_ts(t, window_start, window_end, 0.001), overrides)

    n_gnl_pairs = counts.get("gnl", 0) // 2
    slot_gnl = window_secs / (n_gnl_pairs + 1) if n_gnl_pairs else window_secs
    for i in range(n_gnl_pairs):
        base = window_start + timedelta(seconds=slot_gnl * (i + 1))
        jitter = rng.uniform(-slot_gnl * 0.4, slot_gnl * 0.4)
        t0 = _clamp_ts(base + timedelta(seconds=jitter), window_start, window_end, 5)
        tmpl = rng.choice(prof.gnl_templates)
        inv = str(invoke_counter[0]); invoke_counter[0] += 1
        rtt_val = rng.choice(rtt) if rtt else 0.05

        for direction, offset in [("REQUEST", 0.0), ("RESPONSE", rtt_val)]:
            overrides = {"direction": direction, "service": "GET_NAME_LIST", "invoke_id": inv}
            if direction == "RESPONSE":
                overrides.update({
                    "src_ip": tmpl["dst_ip"],
                    "src_port": tmpl["dst_port"],
                    "dst_ip": tmpl["src_ip"],
                    "dst_port": tmpl["src_port"],
                })
            emit(tmpl, _clamp_ts(t0 + timedelta(seconds=offset), window_start, window_end, 0.001), overrides)

    reports_by_stream: Dict[str, List[dict]] = defaultdict(list)
    for row in prof.report_templates:
        reports_by_stream[row["stream_id"]].append(row)
    report_weights = {
        sid: prof.report_stream_counts.get(sid, len(rows))
        for sid, rows in reports_by_stream.items()
    }
    report_counts = _allocate_counts(counts.get("report", 0), report_weights)
    for sid, n_report_stream in report_counts.items():
        if n_report_stream <= 0:
            continue
        stream_reports = reports_by_stream.get(sid, [])
        if not stream_reports:
            continue
        slot_r = window_secs / (n_report_stream + 1)
        for i in range(n_report_stream):
            base = window_start + timedelta(seconds=slot_r * (i + 1))
            jitter = rng.uniform(-slot_r * 0.3, slot_r * 0.3)
            t = _clamp_ts(base + timedelta(seconds=jitter), window_start, window_end, 0.001)
            emit(rng.choice(stream_reports), t, {"direction": "RESPONSE", "service": "UNCONFIRMED"})

    n_other = counts.get("other", 0)
    slot_other = window_secs / (n_other + 1) if n_other else window_secs
    for i in range(n_other):
        base = window_start + timedelta(seconds=slot_other * (i + 1))
        jitter = rng.uniform(-slot_other * 0.3, slot_other * 0.3)
        t = _clamp_ts(base + timedelta(seconds=jitter), window_start, window_end, 0.001)
        emit(rng.choice(prof.other_templates), t)

    fallback_templates = (
        prof.report_templates
        or prof.other_templates
        or prof.gnl_templates
        or [pair[0] for pair in prof.control_templates]
    )
    n_fallback = counts.get("fallback", 0)
    slot_fallback = window_secs / (n_fallback + 1) if n_fallback else window_secs
    for i in range(n_fallback):
        base = window_start + timedelta(seconds=slot_fallback * (i + 1))
        t = _clamp_ts(base, window_start, window_end, 0.001)
        emit(rng.choice(fallback_templates), t)

    return result


def proportional_attack_count(context: EventContextProfile, n_synth_normal: int) -> int:
    return max(0, round(n_synth_normal * context.attack_per_normal_row))


def _synthetic_oper_response_anchors(rows: List[dict]) -> List[datetime]:
    by_invoke: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("invoke_id"):
            by_invoke[row["invoke_id"]].append(row)

    anchors: List[datetime] = []
    for row in rows:
        if (
            row.get("tag") == NORMAL_TAG
            and row.get("direction") == "REQUEST"
            and row.get("service") == "WRITE"
            and row.get("control_action") == "Oper"
        ):
            responses = [
                r for r in by_invoke.get(row.get("invoke_id", ""), [])
                if r.get("direction") == "RESPONSE" and r.get("timestamp", "") >= row.get("timestamp", "")
            ]
            anchor_row = min(responses, key=lambda r: r.get("timestamp", ""), default=row)
            anchors.append(_ts(anchor_row["timestamp"]))
    anchors.sort()
    return anchors


def plan_proportional_attack_times(
    synth_norm: List[dict],
    n_attacks: int,
    context: EventContextProfile,
    window_start: datetime,
    window_end: datetime,
    rng: random.Random,
    min_attack_gap: float,
) -> List[datetime]:
    if n_attacks <= 0:
        return []

    anchors = _synthetic_oper_response_anchors(synth_norm)
    offsets = context.oper_resp_to_report_deltas or [0.1]

    candidates: List[datetime] = []
    if anchors and n_attacks <= len(anchors):
        step = len(anchors) / n_attacks if n_attacks <= len(anchors) else 1.0
        for i in range(n_attacks):
            anchor = anchors[min(len(anchors) - 1, int(i * step))]
            candidates.append(anchor + timedelta(seconds=float(rng.choice(offsets))))
    elif anchors:
        for anchor in anchors:
            candidates.append(anchor + timedelta(seconds=float(rng.choice(offsets))))

    window_secs = max((window_end - window_start).total_seconds(), float(min_attack_gap) * (n_attacks + 1))
    if not candidates:
        slot = window_secs / (n_attacks + 1)
        jitter = min(slot * 0.35, 300.0)
        candidates = [
            window_start + timedelta(seconds=(slot * (i + 1)) + rng.uniform(-jitter, jitter))
            for i in range(n_attacks)
        ]

    candidates = [_clamp_ts(ts, window_start, window_end, 0.001) for ts in candidates]
    candidates.sort()
    spaced: List[datetime] = []
    for candidate in candidates:
        if spaced and (candidate - spaced[-1]).total_seconds() < min_attack_gap:
            continue
        spaced.append(candidate)
        if len(spaced) >= n_attacks:
            return spaced[:n_attacks]

    attempts = 0
    while len(spaced) < n_attacks and attempts < n_attacks * 200:
        attempts += 1
        candidate = window_start + timedelta(seconds=rng.uniform(0.0, window_secs))
        candidate = _clamp_ts(candidate, window_start, window_end, 0.001)
        if all(abs((candidate - existing).total_seconds()) >= min_attack_gap for existing in spaced):
            spaced.append(candidate)
            spaced.sort()

    cursor = window_start + timedelta(seconds=rng.uniform(0.0, max(float(min_attack_gap), 1.0)))
    while len(spaced) < n_attacks:
        if all(abs((cursor - existing).total_seconds()) >= min_attack_gap for existing in spaced):
            spaced.append(cursor)
            spaced.sort()
        if cursor > window_end and spaced:
            cursor = spaced[-1]
        cursor += timedelta(seconds=float(min_attack_gap) + rng.uniform(1.0, max(float(min_attack_gap), 1.0)))

    return spaced[:n_attacks]


# ── Main ──────────────────────────────────────────────────────────────────────

def plan_isolated_attack_times(
    n_attacks: int,
    window_start: datetime,
    window_end: datetime,
    rng: random.Random,
    min_attack_gap: float,
) -> List[datetime]:
    """Fast jittered schedule for large exact-count attack datasets."""
    if n_attacks <= 0:
        return []

    min_gap = max(float(min_attack_gap), 0.001)
    window_secs = max((window_end - window_start).total_seconds(), min_gap * (n_attacks + 1))
    slot = max(min_gap * 1.2, window_secs / (n_attacks + 1))
    jitter = max(0.0, min((slot - min_gap) * 0.45, slot * 0.25))

    result: List[datetime] = []
    for i in range(n_attacks):
        offset = slot * (i + 1)
        if jitter:
            offset += rng.uniform(-jitter, jitter)
        candidate = window_start + timedelta(seconds=offset)
        if result and (candidate - result[-1]).total_seconds() < min_gap:
            candidate = result[-1] + timedelta(seconds=min_gap + rng.uniform(0.001, max(jitter, 0.001)))
        result.append(candidate)

    return result


def plan_attack_times_from_normal_context(
    synth_norm: List[dict],
    n_attacks: int,
    window_start: datetime,
    window_end: datetime,
    rng: random.Random,
    min_attack_gap: float,
) -> List[datetime]:
    if n_attacks <= 0:
        return []
    if not synth_norm:
        return plan_isolated_attack_times(n_attacks, window_start, window_end, rng, min_attack_gap)

    normal_times = sorted(_ts(row["timestamp"]) for row in synth_norm if row.get("timestamp"))
    if not normal_times:
        return plan_isolated_attack_times(n_attacks, window_start, window_end, rng, min_attack_gap)

    min_gap = max(float(min_attack_gap), 0.001)
    spaced: List[datetime] = []

    idx = 0
    for _ in range(n_attacks):
        candidate = None
        while idx < len(normal_times):
            candidate = normal_times[idx] + timedelta(seconds=rng.uniform(0.02, 0.25))
            idx += 1
            if not spaced or (candidate - spaced[-1]).total_seconds() >= min_gap:
                break
        if candidate is None or (spaced and (candidate - spaced[-1]).total_seconds() < min_gap):
            candidate = spaced[-1] + timedelta(seconds=min_gap + rng.uniform(0.001, min_gap * 0.2))
        spaced.append(candidate)

    return spaced[:n_attacks]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input",  default="data/raw/mms_capture_attack_tags.csv")
    p.add_argument("--output", default="data/raw/mms_capture_augmented.csv.gz")
    p.add_argument("--scale",  type=float, default=2.0,
                   help="Total output size as a multiple of the original (default 2)")
    p.add_argument("--attack-mode", choices=["proportional", "ratio"], default="proportional",
                   help="proportional uses the baseline attack-per-normal rate; ratio uses --ratio")
    p.add_argument("--ratio",  type=float, default=0.40,
                   help="Target attack fraction when --attack-mode ratio is used")
    p.add_argument("--target-attacks", type=int, default=None,
                   help="Exact final attack row count; overrides --attack-mode attack target")
    p.add_argument("--target-normal", type=int, default=None,
                   help="Exact final normal row count")
    p.add_argument("--target-request-response", type=int, default=None,
                   help="Exact final normal request/response row count; normal reports are added separately")
    p.add_argument("--target-normal-reports", type=int, default=None,
                   help="Exact final normal UNCONFIRMED report count; defaults to baseline ratio with --target-request-response")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--min-attack-gap", type=float, default=MIN_SYNTHETIC_ATTACK_GAP_SECONDS,
                   help="Minimum seconds between synthetic attack reports (default 61)")
    p.add_argument("--profile-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[1/5] Reading {args.input} ...")
    with _open(args.input) as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        all_rows = list(reader)

    n_orig = len(all_rows)
    n_orig_atk = sum(1 for r in all_rows if r.get("tag") == ATTACK_TAG)
    n_orig_norm = n_orig - n_orig_atk
    n_orig_rr, n_orig_reports = _normal_subtype_counts(all_rows)
    print(f"      {n_orig:,} original rows  ({n_orig_atk} attack / {n_orig_norm:,} normal)")

    print("[2/5] Profiling attack patterns ...")
    atk_profiles = profile_attacks(all_rows)
    for urcb, p in atk_profiles.items():
        if p.burst_sizes:
            short = urcb.split("/")[0]
            print(f"      {short}: {len(p.burst_sizes)} bursts, "
                  f"size {min(p.burst_sizes)}-{max(p.burst_sizes)}, "
                  f"max_seq={p.max_seq}, "
                  f"{len(p.combo_counts)} data_ref combos")

    print("[3/5] Profiling normal traffic ...")
    norm_profile = profile_normal(all_rows)
    print(f"      Control templates : {len(norm_profile.control_templates)} SBOw/Oper pairs")
    print(f"      GET_NAME_LIST tmpl: {len(norm_profile.gnl_templates)} request templates")
    print(f"      Report templates  : {len(norm_profile.report_templates)} across "
          f"{len(norm_profile.report_stream_counts)} streams")
    print(f"      Other normal tmpl : {len(norm_profile.other_templates)} rare service rows")
    print(f"      Normal composition: "
          f"control={norm_profile.n_control_rows}, "
          f"GNL={norm_profile.n_gnl_rows}, "
          f"reports={norm_profile.n_report_rows}, "
          f"other={norm_profile.n_other_rows}")
    context_profile = profile_event_context(all_rows)
    print(f"      Event map        : {context_profile.normal_control_events} normal control-report events, "
          f"{context_profile.attack_reports} attack reports")
    print(f"      Baseline attack rate: {context_profile.attack_per_normal_row:.6f} per normal packet, "
          f"{context_profile.attack_per_control_event:.2f} per control event")
    print(f"      Oper_resp->report: median={_median(context_profile.oper_resp_to_report_deltas, 0.0):.3f}s, "
          f"context +/-60s normal before/after="
          f"{_mean(context_profile.normal_report_rows_before_60s):.1f}/"
          f"{_mean(context_profile.normal_report_rows_after_60s):.1f}, "
          f"attack before/after="
          f"{_mean(context_profile.attack_rows_before_60s):.1f}/"
          f"{_mean(context_profile.attack_rows_after_60s):.1f}")

    if args.profile_only:
        return

    # ── Calculate targets ─────────────────────────────────────────────────────
    exact_target_mode = (
        args.target_attacks is not None
        or args.target_normal is not None
        or args.target_request_response is not None
        or args.target_normal_reports is not None
    )
    target_request_response = None
    target_normal_reports = None
    if args.target_request_response is not None:
        target_request_response = args.target_request_response
        if target_request_response < n_orig_rr:
            raise ValueError(
                f"--target-request-response={target_request_response} is below the existing "
                f"{n_orig_rr} normal request/response rows"
            )
        if args.target_normal_reports is not None:
            target_normal_reports = args.target_normal_reports
        else:
            report_ratio = n_orig_reports / n_orig_rr if n_orig_rr else 0.0
            target_normal_reports = round(target_request_response * report_ratio)
        if target_normal_reports < n_orig_reports:
            raise ValueError(
                f"--target-normal-reports={target_normal_reports} is below the existing "
                f"{n_orig_reports} normal report rows"
            )

    if exact_target_mode:
        target_attack = args.target_attacks if args.target_attacks is not None else n_orig_atk
        if target_request_response is not None:
            target_normal = target_request_response + target_normal_reports
        else:
            target_normal = args.target_normal if args.target_normal is not None else round(n_orig_norm * args.scale)
        if target_attack < n_orig_atk:
            raise ValueError(f"--target-attacks={target_attack} is below the existing {n_orig_atk} attack rows")
        if target_normal < n_orig_norm:
            raise ValueError(f"--target-normal={target_normal} is below the existing {n_orig_norm} normal rows")
        n_synth_atk = target_attack - n_orig_atk
        if target_request_response is not None:
            n_synth_rr = target_request_response - n_orig_rr
            n_synth_reports = target_normal_reports - n_orig_reports
            n_synth_norm = n_synth_rr + n_synth_reports
        else:
            n_synth_rr = None
            n_synth_reports = None
            n_synth_norm = target_normal - n_orig_norm
        target_total = target_attack + target_normal
    elif args.attack_mode == "ratio":
        target_total  = round(n_orig * args.scale)
        target_attack = round(target_total * args.ratio)
        target_normal = target_total - target_attack
        n_synth_atk   = max(0, target_attack - n_orig_atk)
        n_synth_norm  = max(0, target_normal - n_orig_norm)
    else:
        target_normal = round(n_orig_norm * args.scale)
        n_synth_norm = max(0, target_normal - n_orig_norm)
        n_synth_atk = proportional_attack_count(context_profile, n_synth_norm)
        target_attack = n_orig_atk + n_synth_atk
        target_total = target_normal + target_attack

    mode_label = "exact-target" if exact_target_mode else args.attack_mode
    print(f"\n[4/5] Generating synthetic rows (scale={args.scale}x, attack_mode={mode_label})")
    print(f"      Target total  : {target_total:,}")
    print(f"      Target attack : {target_attack:,}  (generate {n_synth_atk:,})")
    print(f"      Target normal : {target_normal:,}  (generate {n_synth_norm:,})")
    if target_request_response is not None:
        print(f"      Target normal request/response: {target_request_response:,}  "
              f"(generate {n_synth_rr:,})")
        print(f"      Target normal reports         : {target_normal_reports:,}  "
              f"(generate {n_synth_reports:,})")

    all_ts = [_ts(r["timestamp"]) for r in all_rows if r.get("timestamp")]
    orig_start = min(all_ts)
    orig_end   = max(all_ts)
    orig_dur   = (orig_end - orig_start).total_seconds()

    # Extend window proportionally so normal traffic density stays realistic.
    normal_scale = target_normal / n_orig_norm if n_orig_norm else args.scale
    synth_end = orig_end + timedelta(seconds=orig_dur * (max(args.scale, normal_scale) - 1))
    if n_synth_atk > 0 and args.min_attack_gap > 0 and (exact_target_mode or args.attack_mode == "ratio"):
        min_attack_end = orig_end + timedelta(seconds=args.min_attack_gap * (n_synth_atk + 1) * 1.2)
        if min_attack_end > synth_end:
            synth_end = min_attack_end

    base_frame = max(int(r.get("frame_number", 0) or 0) for r in all_rows) + 1
    max_invoke = max(
        (int(r["invoke_id"]) for r in all_rows
         if r.get("invoke_id") and r["invoke_id"].isdigit() and len(r["invoke_id"]) <= 9),
        default=200000
    )
    max_ctl = max(
        (int(r["ctl_num"]) for r in all_rows
         if r.get("tag") == NORMAL_TAG
         and r.get("direction") == "REQUEST"
         and r.get("service") == "WRITE"
         and r.get("ctl_num")
         and r["ctl_num"].isdigit()),
        default=0
    )
    frame_counter  = [base_frame]
    invoke_counter = [max_invoke + 1]
    control_counter = [max_ctl + 1]

    if target_request_response is not None:
        synth_norm = replay_normal_baseline_by_targets(
            all_rows, n_synth_rr, n_synth_reports, orig_start, orig_end,
            orig_end + timedelta(microseconds=1), synth_end,
            frame_counter, invoke_counter, control_counter, fieldnames
        )
    else:
        synth_norm = replay_normal_baseline(
            all_rows, n_synth_norm, orig_start, orig_end,
            frame_counter, invoke_counter, control_counter, fieldnames
        )
    if target_request_response is None and len(synth_norm) < n_synth_norm:
        filler_needed = n_synth_norm - len(synth_norm)
        synth_norm.extend(generate_normal(
            _normal_without_controls(norm_profile), filler_needed, orig_end, synth_end,
            frame_counter, invoke_counter, rng, fieldnames, control_counter
        ))
    print(f"      Replayed {len(synth_norm):,} synthetic normal rows from the baseline timeline")

    attack_times = None
    if exact_target_mode and n_synth_atk > 0:
        attack_times = plan_attack_times_from_normal_context(
            synth_norm, n_synth_atk, orig_end, synth_end, rng, args.min_attack_gap
        )
        if attack_times:
            synth_end = max(synth_end, max(attack_times) + timedelta(milliseconds=1))
    elif args.attack_mode == "proportional":
        attack_times = plan_proportional_attack_times(
            synth_norm, n_synth_atk, context_profile, orig_end, synth_end, rng, args.min_attack_gap
        )
        if attack_times:
            synth_end = max(synth_end, max(attack_times) + timedelta(milliseconds=1))

    synth_atk = generate_attacks(
        atk_profiles, n_synth_atk, orig_end, synth_end, frame_counter, rng,
        args.min_attack_gap, scheduled_times=attack_times
    )
    print(f"      Generated {len(synth_atk):,} synthetic attack rows")

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"[5/5] Writing {args.output} ...")
    merged = all_rows + synth_atk + synth_norm
    merged.sort(key=lambda r: (r.get("timestamp", ""), int(r.get("frame_number") or 0)))
    for frame_number, row in enumerate(merged, start=1):
        row["frame_number"] = str(frame_number)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with _open(args.output, "wt") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    n_out = len(merged)
    n_out_atk = sum(1 for r in merged if r.get("tag") == ATTACK_TAG)
    n_out_norm = n_out - n_out_atk
    print(f"      {n_out:,} total rows")
    print(f"      {n_out_atk:,} attack  ({n_out_atk/n_out:.1%})")
    print(f"      {n_out_norm:,} normal  ({n_out_norm/n_out:.1%})")
    print(f"      Time window: {_ts_str(orig_start)} -> {_ts_str(synth_end)}")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
                                                                                                                                         