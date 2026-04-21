#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

SEVERITY = {"normal": 0, "failed-control": 1, "likely-attack": 2}
CONTROL_HINTS = ("$ST$Pos", "$CO$Pos", "$ST$EnaCls", "CSWI", "CILO")

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid MMS IDS: protocol-state + statistical baseline + unified scoring + suppression."
    )
    parser.add_argument("--input-csv", default="data/sample/mms_sample_100k.csv")
    parser.add_argument("--output-csv", default="hybrid_ids_alerts.csv")
    parser.add_argument("--baseline-json", default="hybrid_ids_baseline.json")
    parser.add_argument("--train-duration-min", type=float, default=5.0)
    parser.add_argument("--fallback-train-rows", type=int, default=50000)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--pair-timeout-sec", type=float, default=15.0)
    parser.add_argument("--report-match-window-sec", type=float, default=20.0)
    parser.add_argument(
        "--strict-report-write-correlation",
        action="store_true",
        help=(
            "If set, always flag report_without_matching_write when no recent write "
            "is found. By default, baseline-known origin + known-ctl reports are "
            "treated as valid delayed/cancel replay context."
        ),
    )
    parser.add_argument(
        "--use-training-control-baseline",
        action="store_true",
        help=(
            "Seed control-origin and ctlNum protocol checks from the training prefix. "
            "By default, these control baselines are learned online after the "
            "training window to avoid prefix contamination."
        ),
    )
    parser.add_argument(
        "--enable-report-seq-check",
        action="store_true",
        help=(
            "Enable monotonicity checks on report sequence numbers extracted from "
            "top-level RPT unsigned values."
        ),
    )
    parser.add_argument(
        "--report-seq-reset-gap-sec",
        type=float,
        default=120.0,
        help=(
            "If time gap between same report stream updates exceeds this, allow "
            "sequence reset without flagging regression."
        ),
    )
    parser.add_argument(
        "--report-seq-reorder-tolerance",
        type=int,
        default=2,
        help=(
            "Allow small backward movement in report sequence (out-of-order capture) "
            "up to this value without flagging regression."
        ),
    )
    parser.add_argument(
        "--report-seq-flag-duplicate",
        action="store_true",
        help="Flag equal consecutive report sequence values as duplicate.",
    )
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--rate-z-threshold", type=float, default=3.0)
    parser.add_argument("--rare-service-prob", type=float, default=0.005)
    parser.add_argument("--min-baseline-samples", type=int, default=20)
    parser.add_argument("--min-rate-samples", type=int, default=5)
    parser.add_argument("--unknown-channel-rate-threshold", type=int, default=25)
    parser.add_argument("--suppress-sec", type=float, default=30.0)
    parser.add_argument("--score-failed-threshold", type=float, default=40.0)
    parser.add_argument("--score-likely-threshold", type=float, default=70.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--include-training-phase", action="store_true")
    parser.add_argument("--emit-all", action="store_true")
    return parser.parse_args()


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


def channel_key(src_ip: str, dst_ip: str, direction: str, service: str) -> str:
    return f"{src_ip}|{dst_ip}|{direction}|{service}"


def payload_size_bytes(raw_hex: str) -> Optional[int]:
    if not raw_hex:
        return None
    s = raw_hex.strip()
    if not s or len(s) % 2 != 0:
        return None
    return len(s) // 2


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from iter_strings(v)
        return
    if isinstance(value, list):
        for v in value:
            yield from iter_strings(v)


def iter_typed_string_values(value: Any, expected_type: str) -> Iterable[str]:
    if isinstance(value, dict):
        if value.get("type") == expected_type and isinstance(value.get("value"), str):
            yield value["value"]
        for v in value.values():
            yield from iter_typed_string_values(v, expected_type)
        return
    if isinstance(value, list):
        for v in value:
            yield from iter_typed_string_values(v, expected_type)


def iter_typed_numeric_values(value: Any, expected_type: str) -> Iterable[int]:
    if isinstance(value, dict):
        raw = value.get("value")
        if value.get("type") == expected_type and isinstance(raw, int):
            yield raw
        for v in value.values():
            yield from iter_typed_numeric_values(v, expected_type)
        return
    if isinstance(value, list):
        for v in value:
            yield from iter_typed_numeric_values(v, expected_type)


def contains_control_hint(raw: str) -> bool:
    return bool(raw) and any(token in raw for token in CONTROL_HINTS)


def extract_control_refs(variables: List[Any], access_result: List[Any]) -> List[str]:
    refs: set[str] = set()
    for s in iter_strings(variables):
        if any(token in s for token in CONTROL_HINTS):
            refs.add(s)
    for item in access_result:
        if isinstance(item, dict) and item.get("type") == "visible-string":
            val = item.get("value")
            if isinstance(val, str) and any(token in val for token in CONTROL_HINTS):
                refs.add(val)
    return sorted(refs)


def extract_report_control_entries(variables: List[Any], access_result: List[Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    refs = extract_control_refs(variables, access_result)
    seen: set[Tuple[str, Optional[int]]] = set()
    for item in access_result:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "structure" or not isinstance(item.get("value"), list):
            continue
        origins = sorted(set(iter_typed_string_values(item, "octet-string")))
        if not origins:
            continue
        unsigned_values = list(iter_typed_numeric_values(item, "unsigned"))
        ctl = unsigned_values[0] if unsigned_values else None
        for origin in origins:
            key = (origin, ctl)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"origin_identifier": origin, "ctl_num": ctl, "control_refs": refs})
    return entries


def extract_report_sequence_meta(access_result: List[Any]) -> Tuple[str, Optional[int]]:
    report_name: str = ""
    first_visible: str = ""
    report_seq: Optional[int] = None

    for item in access_result:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        value = item.get("value")
        if item_type == "visible-string" and isinstance(value, str):
            if not first_visible:
                first_visible = value
            if not report_name and (
                "$RP$" in value or ".urcb" in value.lower() or "urcb" in value.lower() or "brcb" in value.lower()
            ):
                report_name = value
        elif item_type == "unsigned" and isinstance(value, int) and report_seq is None:
            # Top-level unsigned in RPT payload acts as report sequence counter.
            report_seq = value

        if report_name and report_seq is not None:
            break

    if not report_name:
        report_name = first_visible
    return report_name, report_seq


def extract_boolean_values(value: Any) -> List[str]:
    values: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") == "boolean" and isinstance(x.get("value"), bool):
                values.append("true" if x["value"] else "false")
            for y in x.values():
                walk(y)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(value)
    return values


def build_alert_fingerprint(
    *,
    final_tag: str,
    src_ip: str,
    dst_ip: str,
    service: str,
    reasons: List[str],
    report_origins: List[str],
    report_ctl_nums: List[int],
    report_seq_num: Optional[int],
    report_control_refs: List[str],
    report_boolean_values: List[str],
) -> str:
    reason_key = ";".join(sorted(set(reasons))[:3])
    origin_key = ";".join(report_origins)
    ctl_key = ";".join(str(v) for v in report_ctl_nums)
    seq_key = str(report_seq_num) if isinstance(report_seq_num, int) else ""
    ref_key = ";".join(report_control_refs)
    bool_key = ";".join(report_boolean_values)
    return (
        f"{final_tag}|{src_ip}|{dst_ip}|{service}|{reason_key}|"
        f"{origin_key}|{ctl_key}|{seq_key}|{ref_key}|{bool_key}"
    )


def upgrade_tag(current_tag: str, reasons: List[str], new_tag: str, reason: str) -> Tuple[str, List[str]]:
    out_tag = current_tag
    if SEVERITY[new_tag] > SEVERITY[out_tag]:
        out_tag = new_tag
    if reason and reason not in reasons:
        reasons.append(reason)
    return out_tag, reasons

def train_baseline(input_csv: Path, args: argparse.Namespace) -> Dict[str, Any]:
    service_counts: Counter[str] = Counter()
    interarrival_stats: Dict[str, RunningStats] = defaultdict(RunningStats)
    size_stats: Dict[str, RunningStats] = defaultdict(RunningStats)
    rate_bucket_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    last_ts_by_channel: Dict[str, datetime] = {}

    write_origin_counts: Counter[str] = Counter()
    write_ctlnums_by_origin: Dict[str, set[int]] = defaultdict(set)

    first_dt: Optional[datetime] = None
    train_end_dt: Optional[datetime] = None
    trained_rows = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for _, row in enumerate(reader, start=1):
            ts_raw = row.get("wtimestamp") or row.get("timestamp") or ""
            dt = parse_iso(ts_raw)
            if first_dt is None and dt is not None:
                first_dt = dt
                train_end_dt = first_dt + timedelta(minutes=args.train_duration_min)

            if train_end_dt is not None and dt is not None:
                in_train = dt <= train_end_dt
            else:
                in_train = trained_rows < args.fallback_train_rows

            if not in_train:
                if trained_rows >= args.min_train_rows:
                    break
                continue

            trained_rows += 1
            if trained_rows % 200000 == 0:
                print(f"[train] processed {trained_rows} rows")

            direction = (row.get("direction") or "").strip()
            service = (row.get("service") or "").strip()
            src_ip = (row.get("src_ip") or "").strip()
            dst_ip = (row.get("dst_ip") or "").strip()
            key = channel_key(src_ip, dst_ip, direction, service)

            service_counts[f"{direction}|{service}"] += 1

            raw_hex = (row.get("raw_mms_hex") or "").strip()
            size = payload_size_bytes(raw_hex)
            if size is not None:
                size_stats[key].add(float(size))

            if dt is not None:
                prev = last_ts_by_channel.get(key)
                if prev is not None:
                    delta = (dt - prev).total_seconds()
                    if 0.0 <= delta <= 3600.0:
                        interarrival_stats[key].add(delta)
                last_ts_by_channel[key] = dt
                bucket = int(dt.timestamp() // args.window_sec)
                rate_bucket_counts[key][bucket] += 1

            is_control = parse_int(row.get("is_control")) == 1
            is_control_write = direction == "REQUEST" and service == "WRITE" and is_control
            if is_control_write:
                cp = safe_json_loads(row.get("control_parameters"), {})
                if not isinstance(cp, dict):
                    cp = {}
                origin = ((cp.get("originIdentifier") or {}).get("value"))
                ctl_num = ((cp.get("ctlNum") or {}).get("value"))
                if isinstance(origin, str) and origin:
                    write_origin_counts[origin] += 1
                    if isinstance(ctl_num, int):
                        write_ctlnums_by_origin[origin].add(ctl_num)

    service_total = sum(service_counts.values())
    service_probs = {
        key: (count / service_total)
        for key, count in service_counts.items()
        if service_total > 0
    }

    rate_stats: Dict[str, Dict[str, float]] = {}
    for key, bucket_counts in rate_bucket_counts.items():
        rs = RunningStats()
        for count in bucket_counts.values():
            rs.add(float(count))
        rate_stats[key] = rs.to_dict()

    allowed_origin: Optional[str] = None
    if write_origin_counts:
        allowed_origin = write_origin_counts.most_common(1)[0][0]

    return {
        "trained_rows": trained_rows,
        "train_start": first_dt.isoformat() if first_dt else "",
        "train_end": train_end_dt.isoformat() if train_end_dt else "",
        "window_sec": float(args.window_sec),
        "service_probs": service_probs,
        "interarrival_stats": {k: v.to_dict() for k, v in interarrival_stats.items()},
        "size_stats": {k: v.to_dict() for k, v in size_stats.items()},
        "rate_stats": rate_stats,
        "allowed_origin": allowed_origin,
        "known_origins": sorted(write_origin_counts.keys()),
        "write_ctlnums_by_origin": {k: sorted(v) for k, v in write_ctlnums_by_origin.items()},
    }


def compute_statistical_score(
    event: Dict[str, Any],
    model: Dict[str, Any],
    args: argparse.Namespace,
    last_ts_by_channel: Dict[str, datetime],
    live_rate_windows: Dict[str, Deque[datetime]],
    enable_scoring: bool,
) -> Tuple[List[str], float]:
    reasons: List[str] = []
    score = 0.0

    key = channel_key(event["src_ip"], event["dst_ip"], event["direction"], event["service"])
    dt = event["dt"]

    if dt is not None:
        prev_dt = last_ts_by_channel.get(key)
        delta = None
        if prev_dt is not None:
            delta = (dt - prev_dt).total_seconds()
        last_ts_by_channel[key] = dt
    else:
        delta = None

    if dt is not None:
        window_sec = float(model.get("window_sec", args.window_sec))
        dq = live_rate_windows[key]
        cutoff = dt - timedelta(seconds=window_sec)
        while dq and dq[0] < cutoff:
            dq.popleft()
        dq.append(dt)
        current_rate = len(dq)
    else:
        current_rate = 0

    if not enable_scoring:
        return reasons, score

    inter_stats = model.get("interarrival_stats", {}).get(key)
    if (
        inter_stats
        and delta is not None
        and delta >= 0
        and inter_stats.get("n", 0) >= args.min_baseline_samples
        and inter_stats.get("std", 0.0) > 0
    ):
        z = abs(delta - inter_stats["mean"]) / inter_stats["std"]
        if z >= args.z_threshold:
            reasons.append("interarrival_outlier")
            score += min(25.0, 5.0 + (z - args.z_threshold) * 4.0)

    size_stats = model.get("size_stats", {}).get(key)
    if (
        size_stats
        and event["payload_size"] is not None
        and size_stats.get("n", 0) >= args.min_baseline_samples
        and size_stats.get("std", 0.0) > 0
    ):
        z = abs(event["payload_size"] - size_stats["mean"]) / size_stats["std"]
        if z >= args.z_threshold:
            reasons.append("payload_size_outlier")
            score += min(20.0, 4.0 + (z - args.z_threshold) * 3.0)

    service_key = f"{event['direction']}|{event['service']}"
    p = float(model.get("service_probs", {}).get(service_key, 0.0))
    if p <= args.rare_service_prob:
        reasons.append("rare_service_combo")
        rarity = -math.log(max(p, 1e-9))
        score += min(20.0, 4.0 + rarity * 3.0)

    rate_stats = model.get("rate_stats", {}).get(key)
    if rate_stats and rate_stats.get("n", 0) >= args.min_rate_samples and current_rate > 0:
        std = max(float(rate_stats.get("std", 0.0)), 0.5)
        z = (current_rate - float(rate_stats.get("mean", 0.0))) / std
        if z >= args.rate_z_threshold:
            reasons.append("rate_spike")
            score += min(25.0, 5.0 + (z - args.rate_z_threshold) * 4.0)
    elif current_rate >= args.unknown_channel_rate_threshold:
        reasons.append("unknown_channel_rate_spike")
        score += 12.0

    return reasons, min(score, 100.0)

def detect_with_hybrid_ids(
    input_csv: Path, output_csv: Path, model: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, int]:
    trained_known_origins = set(model.get("known_origins", []))
    trained_write_ctlnums = {k: set(v) for k, v in model.get("write_ctlnums_by_origin", {}).items()}
    active_known_origins = set(trained_known_origins) if args.use_training_control_baseline else set()
    active_write_ctlnums: Dict[str, set[int]] = defaultdict(set)
    if args.use_training_control_baseline:
        active_write_ctlnums.update(trained_write_ctlnums)
    active_allowed_origin = model.get("allowed_origin") if args.use_training_control_baseline else None
    active_write_origin_counts: Counter[str] = Counter()
    if isinstance(active_allowed_origin, str) and active_allowed_origin:
        active_write_origin_counts[active_allowed_origin] = 1
    train_end_dt = parse_iso(model.get("train_end", ""))

    pending_sbow: Dict[Tuple[str, str, str, Any], Dict[str, Any]] = {}
    last_write_invoke: Dict[Tuple[str, str], int] = {}
    seen_write_invoke: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(dict)
    last_write_ctl_by_origin: Dict[Tuple[str, str, str], int] = {}
    ctl_usage: Dict[Tuple[str, str, str, int], Dict[str, Any]] = defaultdict(lambda: {"count": 0, "objects": set()})
    report_last_ctl_by_origin: Dict[Tuple[str, str, str], int] = {}
    report_last_seq_by_stream: Dict[Tuple[str, str, str, str, str], Tuple[int, Optional[datetime]]] = {}
    recent_writes: Dict[Tuple[str, str, str, int], Deque[datetime]] = defaultdict(deque)

    stat_last_ts_by_channel: Dict[str, datetime] = {}
    stat_live_rates: Dict[str, Deque[datetime]] = defaultdict(deque)
    suppression_last_sent: Dict[str, datetime] = {}

    totals: Counter[str] = Counter()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_columns = [
        "event_source", "line_number", "timestamp", "direction", "service", "src_ip", "dst_ip",
        "stream_id", "invoke_id", "control_object", "control_action", "ctl_num", "origin_identifier",
        "report_seq_num", "report_origins", "report_ctl_nums", "report_control_refs", "report_boolean_values",
        "protocol_tag", "protocol_reasons", "stat_reasons",
        "protocol_score", "stat_score", "anomaly_score", "final_tag", "suppressed", "summary",
    ]

    with input_csv.open("r", encoding="utf-8", newline="") as in_f, output_csv.open("w", encoding="utf-8", newline="") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=out_columns)
        writer.writeheader()

        for line_number, row in enumerate(reader, start=1):
            if args.max_rows and line_number > args.max_rows:
                break
            totals["rows_seen"] += 1
            if totals["rows_seen"] % 200000 == 0:
                print(f"[detect] processed {totals['rows_seen']} rows")

            ts_raw = (row.get("wtimestamp") or row.get("timestamp") or "").strip()
            dt = parse_iso(ts_raw)
            direction = (row.get("direction") or "").strip()
            service = (row.get("service") or "").strip()
            src_ip = (row.get("src_ip") or "").strip()
            dst_ip = (row.get("dst_ip") or "").strip()
            stream_id = (row.get("stream_id") or "").strip()
            variable_list_name = (row.get("variable_list_name") or "").strip()
            summary = (row.get("summary") or "").strip()
            control_object = (row.get("control_object") or "").strip()
            control_action = (row.get("control_action") or "").strip()
            invoke_id = parse_int(row.get("invoke_id"))
            is_control = parse_int(row.get("is_control")) == 1

            payload_size = payload_size_bytes((row.get("raw_mms_hex") or "").strip())
            cp = safe_json_loads(row.get("control_parameters"), {})
            if not isinstance(cp, dict):
                cp = {}
            ctl_num = ((cp.get("ctlNum") or {}).get("value"))
            origin_identifier = ((cp.get("originIdentifier") or {}).get("value"))

            variables_raw = row.get("variables") or ""
            access_raw = row.get("access_result") or ""
            control_context = is_control or contains_control_hint(variables_raw) or contains_control_hint(access_raw)
            is_report_packet = service == "UNCONFIRMED" and (
                variable_list_name == "RPT" or "<RPT>" in summary
            )

            parsed_variables: Optional[List[Any]] = None
            parsed_access_result: Optional[List[Any]] = None

            detection_enabled = True
            if (not args.include_training_phase and train_end_dt is not None and dt is not None and dt <= train_end_dt):
                detection_enabled = False

            event = {
                "line_number": line_number,
                "timestamp": ts_raw,
                "dt": dt,
                "direction": direction,
                "service": service,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "stream_id": stream_id,
                "summary": summary,
                "invoke_id": invoke_id,
                "control_object": control_object,
                "control_action": control_action,
                "ctl_num": ctl_num,
                "origin_identifier": origin_identifier,
                "payload_size": payload_size,
                "report_seq_num": None,
            }

            protocol_tag = "normal"
            protocol_reasons: List[str] = []
            report_entries: List[Dict[str, Any]] = []
            report_seq_name: str = ""
            report_control_refs: List[str] = []
            report_boolean_values: List[str] = []

            if "LastApplError" in str(variables_raw):
                protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "last_appl_error")

            if is_report_packet:
                parsed_access_result = safe_json_loads(access_raw, [])
                if not isinstance(parsed_access_result, list):
                    parsed_access_result = []
                report_name, report_seq = extract_report_sequence_meta(parsed_access_result)
                if not report_name:
                    report_name = variable_list_name or "RPT"
                report_seq_name = report_name
                event["report_seq_num"] = report_seq

            is_control_request = direction == "REQUEST" and service == "WRITE" and is_control
            if is_control_request:
                key = (src_ip, dst_ip, control_object, ctl_num)
                if control_action == "SBOw":
                    pending_sbow[key] = {
                        "line_number": line_number,
                        "timestamp": ts_raw,
                        "dt": dt,
                        "direction": direction,
                        "service": service,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "stream_id": stream_id,
                        "invoke_id": invoke_id,
                        "control_object": control_object,
                        "control_action": control_action,
                        "ctl_num": ctl_num,
                        "origin_identifier": origin_identifier,
                        "summary": summary,
                    }
                elif control_action == "Oper":
                    if key not in pending_sbow:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "oper_without_matching_sbow")
                    else:
                        sbow_event = pending_sbow.pop(key)
                        sbow_dt = sbow_event.get("dt")
                        if sbow_dt is not None and dt is not None:
                            delay = (dt - sbow_dt).total_seconds()
                            if delay > args.pair_timeout_sec:
                                protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", f"late_oper_after_{delay:.3f}s")

                channel = (src_ip, dst_ip)
                if invoke_id is None:
                    protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "missing_invoke_id_on_write")
                else:
                    last_invoke = last_write_invoke.get(channel)
                    if last_invoke is not None and invoke_id < last_invoke:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "write_invoke_id_regression")
                    seen_map = seen_write_invoke[channel]
                    if invoke_id in seen_map:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "duplicate_write_invoke_id")
                    seen_map[invoke_id] = line_number
                    if len(seen_map) > 5000:
                        oldest_key = next(iter(seen_map))
                        seen_map.pop(oldest_key, None)
                    if last_invoke is None or invoke_id >= last_invoke:
                        last_write_invoke[channel] = invoke_id

                origin_key = origin_identifier if isinstance(origin_identifier, str) and origin_identifier else "UNKNOWN"
                if isinstance(ctl_num, int):
                    ctl_seq_key = (src_ip, dst_ip, origin_key)
                    prev_ctl = last_write_ctl_by_origin.get(ctl_seq_key)
                    if prev_ctl is not None and ctl_num < prev_ctl:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "write_ctlnum_regression")
                    if prev_ctl is None or ctl_num >= prev_ctl:
                        last_write_ctl_by_origin[ctl_seq_key] = ctl_num

                    usage_key = (src_ip, dst_ip, origin_key, ctl_num)
                    usage = ctl_usage[usage_key]
                    usage["count"] += 1
                    if control_object:
                        usage["objects"].add(control_object)
                    if len(usage["objects"]) > 1:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "write_ctlnum_reused_across_objects")
                    if usage["count"] > 2:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "write_ctlnum_reused_excessively")

                if (
                    active_allowed_origin
                    and isinstance(origin_identifier, str)
                    and origin_identifier
                    and origin_identifier != active_allowed_origin
                ):
                    protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "unexpected_origin_identifier")

                if detection_enabled and isinstance(origin_identifier, str) and origin_identifier:
                    active_known_origins.add(origin_identifier)
                    active_write_origin_counts[origin_identifier] += 1
                    if active_allowed_origin is None:
                        active_allowed_origin = origin_identifier
                    else:
                        current_allowed_count = active_write_origin_counts.get(active_allowed_origin, 0)
                        if active_write_origin_counts[origin_identifier] > current_allowed_count:
                            active_allowed_origin = origin_identifier
                    if isinstance(ctl_num, int):
                        active_write_ctlnums[origin_identifier].add(ctl_num)

                if isinstance(origin_identifier, str) and isinstance(ctl_num, int) and dt is not None:
                    write_key = (src_ip, dst_ip, origin_identifier, ctl_num)
                    dq = recent_writes[write_key]
                    dq.append(dt)
                    stale_cutoff = dt - timedelta(seconds=args.report_match_window_sec * 3.0)
                    while dq and dq[0] < stale_cutoff:
                        dq.popleft()

            if control_context:
                if parsed_variables is None:
                    parsed_variables = safe_json_loads(variables_raw, [])
                    if not isinstance(parsed_variables, list):
                        parsed_variables = []
                if parsed_access_result is None:
                    parsed_access_result = safe_json_loads(access_raw, [])
                    if not isinstance(parsed_access_result, list):
                        parsed_access_result = []
                variables = parsed_variables
                access_result = parsed_access_result
                report_control_refs = extract_control_refs(variables, access_result)
                report_boolean_values = extract_boolean_values(access_result)

                octet_identities = sorted(set(iter_typed_string_values([access_result, cp], "octet-string")))
                if active_known_origins:
                    unexpected = [o for o in octet_identities if o not in active_known_origins]
                    if unexpected:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "unexpected_octet_identity_in_control_context")

                report_entries = extract_report_control_entries(variables, access_result)
                if (
                    args.enable_report_seq_check
                    and isinstance(event["report_seq_num"], int)
                    and report_entries
                ):
                    seq_name = report_seq_name or variable_list_name or "RPT"
                    seen_seq_keys: set[Tuple[str, str, str, str, str]] = set()
                    for entry in report_entries:
                        entry_origin = entry.get("origin_identifier")
                        if not isinstance(entry_origin, str) or not entry_origin:
                            entry_origin = "UNKNOWN"
                        entry_refs = entry.get("control_refs") or []
                        seq_scope = ";".join(sorted(entry_refs)) if entry_refs else "*"
                        seq_key = (src_ip, dst_ip, seq_name, entry_origin, seq_scope)
                        if seq_key in seen_seq_keys:
                            continue
                        seen_seq_keys.add(seq_key)

                        prev_state = report_last_seq_by_stream.get(seq_key)
                        prev_seq: Optional[int] = None
                        prev_dt: Optional[datetime] = None
                        if prev_state is not None:
                            prev_seq, prev_dt = prev_state

                        reset_gap = False
                        if prev_dt is not None and dt is not None:
                            reset_gap = (dt - prev_dt).total_seconds() > args.report_seq_reset_gap_sec

                        current_seq = event["report_seq_num"]
                        if prev_seq is not None and not reset_gap:
                            if current_seq < prev_seq:
                                backward_delta = prev_seq - current_seq
                                if backward_delta > args.report_seq_reorder_tolerance:
                                    protocol_tag, protocol_reasons = upgrade_tag(
                                        protocol_tag,
                                        protocol_reasons,
                                        "failed-control",
                                        "report_seq_regression",
                                    )
                            elif current_seq == prev_seq and args.report_seq_flag_duplicate:
                                protocol_tag, protocol_reasons = upgrade_tag(
                                    protocol_tag,
                                    protocol_reasons,
                                    "failed-control",
                                    "report_seq_duplicate",
                                )

                        if prev_seq is None or reset_gap:
                            report_last_seq_by_stream[seq_key] = (current_seq, dt)
                        elif current_seq >= prev_seq:
                            report_last_seq_by_stream[seq_key] = (current_seq, dt)
                        else:
                            # Keep highest seen sequence during tolerated out-of-order captures.
                            report_last_seq_by_stream[seq_key] = (prev_seq, dt)

                for entry in report_entries:
                    report_origin = entry.get("origin_identifier")
                    report_ctl = entry.get("ctl_num")

                    if isinstance(report_origin, str) and report_origin and active_known_origins and report_origin not in active_known_origins:
                        protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "report_origin_not_seen_in_writes")

                    if isinstance(report_origin, str) and report_origin and isinstance(report_ctl, int):
                        rep_seq_key = (src_ip, dst_ip, report_origin)
                        prev_rep_ctl = report_last_ctl_by_origin.get(rep_seq_key)
                        if prev_rep_ctl is not None and report_ctl < prev_rep_ctl:
                            protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "report_ctlnum_regression")
                        if prev_rep_ctl is None or report_ctl >= prev_rep_ctl:
                            report_last_ctl_by_origin[rep_seq_key] = report_ctl

                        expected_ctlnums = active_write_ctlnums.get(report_origin)
                        if expected_ctlnums:
                            min_seen = min(expected_ctlnums)
                            max_seen = max(expected_ctlnums)
                            if report_ctl < min_seen:
                                protocol_tag, protocol_reasons = upgrade_tag(
                                    protocol_tag,
                                    protocol_reasons,
                                    "failed-control",
                                    "report_ctlnum_below_write_baseline",
                                )
                            elif report_ctl <= max_seen and report_ctl not in expected_ctlnums:
                                protocol_tag, protocol_reasons = upgrade_tag(
                                    protocol_tag,
                                    protocol_reasons,
                                    "failed-control",
                                    "report_ctlnum_not_seen_in_writes",
                                )

                        matched = False
                        if dt is not None:
                            candidate_key = (dst_ip, src_ip, report_origin, report_ctl)
                            dq = recent_writes.get(candidate_key, deque())
                            if dq:
                                stale_cutoff = dt - timedelta(seconds=args.report_match_window_sec * 3.0)
                                while dq and dq[0] < stale_cutoff:
                                    dq.popleft()
                                for write_dt in reversed(dq):
                                    delta = (dt - write_dt).total_seconds()
                                    if delta < 0:
                                        continue
                                    if delta <= args.report_match_window_sec:
                                        matched = True
                                        break
                                    if delta > args.report_match_window_sec:
                                        break

                        if not matched:
                            if active_known_origins and report_origin not in active_known_origins:
                                protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "likely-attack", "report_without_matching_write")
                            elif (
                                not args.strict_report_write_correlation
                                and expected_ctlnums
                                and report_ctl in expected_ctlnums
                            ):
                                # Known origin + known ctlNum can legitimately appear as
                                # delayed status/cancel replay without a fresh WRITE.
                                pass
                            else:
                                protocol_tag, protocol_reasons = upgrade_tag(protocol_tag, protocol_reasons, "failed-control", "report_without_matching_write")

            stat_reasons, stat_score = compute_statistical_score(
                event=event,
                model=model,
                args=args,
                last_ts_by_channel=stat_last_ts_by_channel,
                live_rate_windows=stat_live_rates,
                enable_scoring=detection_enabled,
            )

            protocol_base_score = {"normal": 0.0, "failed-control": 45.0, "likely-attack": 70.0}[protocol_tag]
            protocol_bonus = min(20.0, 5.0 * max(0, len(protocol_reasons) - 1))
            protocol_score = protocol_base_score + protocol_bonus
            anomaly_score = min(100.0, protocol_score + stat_score)

            final_tag = "normal"
            if protocol_tag == "likely-attack" or anomaly_score >= args.score_likely_threshold:
                final_tag = "likely-attack"
            elif protocol_tag == "failed-control" or anomaly_score >= args.score_failed_threshold:
                final_tag = "failed-control"

            if not detection_enabled:
                totals["training_phase_rows"] += 1
                continue

            all_reasons = sorted(set(protocol_reasons + stat_reasons))
            report_origins = sorted({str(entry["origin_identifier"]) for entry in report_entries if entry.get("origin_identifier")})
            report_ctl_nums = sorted({int(entry["ctl_num"]) for entry in report_entries if isinstance(entry.get("ctl_num"), int)})
            fingerprint = build_alert_fingerprint(
                final_tag=final_tag,
                src_ip=src_ip,
                dst_ip=dst_ip,
                service=service,
                reasons=all_reasons,
                report_origins=report_origins,
                report_ctl_nums=report_ctl_nums,
                report_seq_num=event["report_seq_num"] if isinstance(event["report_seq_num"], int) else None,
                report_control_refs=report_control_refs,
                report_boolean_values=report_boolean_values,
            )

            suppressed = False
            if final_tag != "normal" and dt is not None and args.suppress_sec > 0:
                last_sent = suppression_last_sent.get(fingerprint)
                if last_sent is not None and (dt - last_sent).total_seconds() < args.suppress_sec:
                    suppressed = True
                else:
                    suppression_last_sent[fingerprint] = dt

            out_row = {
                "event_source": "packet",
                "line_number": line_number,
                "timestamp": ts_raw,
                "direction": direction,
                "service": service,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "stream_id": stream_id,
                "invoke_id": invoke_id if invoke_id is not None else "",
                "control_object": control_object,
                "control_action": control_action,
                "ctl_num": ctl_num if isinstance(ctl_num, int) else "",
                "origin_identifier": origin_identifier if isinstance(origin_identifier, str) else "",
                "report_seq_num": event["report_seq_num"] if isinstance(event["report_seq_num"], int) else "",
                "report_origins": ";".join(report_origins),
                "report_ctl_nums": ";".join(str(v) for v in report_ctl_nums),
                "report_control_refs": ";".join(report_control_refs),
                "report_boolean_values": ";".join(report_boolean_values),
                "protocol_tag": protocol_tag,
                "protocol_reasons": ";".join(protocol_reasons),
                "stat_reasons": ";".join(stat_reasons),
                "protocol_score": f"{protocol_score:.2f}",
                "stat_score": f"{stat_score:.2f}",
                "anomaly_score": f"{anomaly_score:.2f}",
                "final_tag": final_tag,
                "suppressed": int(suppressed),
                "summary": summary,
            }

            totals[f"final_{final_tag}"] += 1
            if suppressed:
                totals["suppressed"] += 1

            should_write = args.emit_all or (final_tag != "normal" and not suppressed)
            if should_write:
                writer.writerow(out_row)
                totals["written_rows"] += 1

        for pending in pending_sbow.values():
            if args.max_rows and pending["line_number"] > args.max_rows:
                continue
            ts_raw = pending.get("timestamp", "")
            dt = pending.get("dt")
            final_tag = "failed-control"
            protocol_reasons = ["sbow_without_matching_oper"]
            protocol_score = 45.0
            stat_score = 0.0
            anomaly_score = protocol_score

            fingerprint = (
                f"{final_tag}|{pending.get('src_ip','')}|{pending.get('dst_ip','')}|"
                f"{pending.get('service','')}|sbow_without_matching_oper|"
                f"{pending.get('control_object','')}|{pending.get('control_action','')}|"
                f"{pending.get('ctl_num','')}|{pending.get('origin_identifier','')}"
            )
            suppressed = False
            if final_tag != "normal" and dt is not None and args.suppress_sec > 0:
                last_sent = suppression_last_sent.get(fingerprint)
                if last_sent is not None and (dt - last_sent).total_seconds() < args.suppress_sec:
                    suppressed = True
                else:
                    suppression_last_sent[fingerprint] = dt

            out_row = {
                "event_source": "synthetic",
                "line_number": pending.get("line_number", ""),
                "timestamp": ts_raw,
                "direction": pending.get("direction", ""),
                "service": pending.get("service", ""),
                "src_ip": pending.get("src_ip", ""),
                "dst_ip": pending.get("dst_ip", ""),
                "stream_id": pending.get("stream_id", ""),
                "invoke_id": pending.get("invoke_id", ""),
                "control_object": pending.get("control_object", ""),
                "control_action": pending.get("control_action", ""),
                "ctl_num": pending.get("ctl_num", ""),
                "origin_identifier": pending.get("origin_identifier", ""),
                "report_seq_num": "",
                "report_origins": "",
                "report_ctl_nums": "",
                "report_control_refs": "",
                "report_boolean_values": "",
                "protocol_tag": "failed-control",
                "protocol_reasons": ";".join(protocol_reasons),
                "stat_reasons": "",
                "protocol_score": f"{protocol_score:.2f}",
                "stat_score": f"{stat_score:.2f}",
                "anomaly_score": f"{anomaly_score:.2f}",
                "final_tag": final_tag,
                "suppressed": int(suppressed),
                "summary": "Pending SBOw without matching Oper",
            }

            totals["pending_sbow"] += 1
            totals["final_failed-control"] += 1
            if suppressed:
                totals["suppressed"] += 1

            should_write = args.emit_all or (final_tag != "normal" and not suppressed)
            if should_write:
                writer.writerow(out_row)
                totals["written_rows"] += 1

    return dict(totals)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    baseline_json = Path(args.baseline_json)

    print(f"[phase-1/2] training baseline from: {input_csv}")
    model = train_baseline(input_csv, args)
    baseline_json.parent.mkdir(parents=True, exist_ok=True)
    baseline_json.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"[baseline] trained_rows={model.get('trained_rows', 0)}")
    print(f"[baseline] train_start={model.get('train_start', '')}")
    print(f"[baseline] train_end={model.get('train_end', '')}")
    print(f"[baseline] known_origins={len(model.get('known_origins', []))}")
    print(f"[baseline] channels={len(model.get('interarrival_stats', {}))}")
    print(f"[baseline] saved={baseline_json}")

    print(f"[phase-3/4] detecting anomalies with suppression to: {output_csv}")
    totals = detect_with_hybrid_ids(input_csv, output_csv, model, args)

    print("[done]")
    print(f"rows_seen: {totals.get('rows_seen', 0)}")
    print(f"training_phase_rows: {totals.get('training_phase_rows', 0)}")
    print(f"final_normal: {totals.get('final_normal', 0)}")
    print(f"final_failed-control: {totals.get('final_failed-control', 0)}")
    print(f"final_likely-attack: {totals.get('final_likely-attack', 0)}")
    print(f"suppressed: {totals.get('suppressed', 0)}")
    print(f"pending_sbow_synthetic: {totals.get('pending_sbow', 0)}")
    print(f"written_rows: {totals.get('written_rows', 0)}")


if __name__ == "__main__":
    main()
