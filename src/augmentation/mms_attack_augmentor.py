"""
MMS IEC 61850 Attack Dataset Augmentation Framework
=====================================================

Expands the labelled dataset (mms_capture_attack_tags.csv) to a target
attack/normal ratio by generating synthetic attack rows that faithfully
reproduce the statistical, temporal, and protocol-specific properties of
the 105 ground-truth attack packets.

Feature Taxonomy (10 features, network → control)
--------------------------------------------------
Network Layer:
  F1. src_ip           Source IED IP (10.0.19.47 or 10.0.19.49)
  F2. dst_ip           SCADA monitor IP (10.0.19.39, constant)
  F3. stream_id        TCP association identifier (maps 1-to-1 with URCB)

Protocol Layer:
  F4. service          MMS service class → UNCONFIRMED (Information Report)
  F5. direction        Packet flow → RESPONSE (IED to SCADA)
  F6. variable_list_name  Report type → RPT (Unbuffered Report Control)

Authorization Layer:
  F7. octet_identities  Client identity embedded in report payload
                         IEDEXPLORER = unauthorized (attack signature)

IEC 61850 Control Layer:
  F8. urcb_ref         Report Control Block reference (access_result[0])
  F9. report_seq_num   Monotonic report sequence per URCB (access_result unsigned)
  F10. data_refs       Control object data references (access_result visible-strings[2:])

Augmentation Method: Deterministic Isolated-Report Synthesis
-------------------------------------------------------------
1. Profile: parse all real attack rows into per-URCB burst profiles capturing
   burst-size distribution, intra-burst inter-arrival distribution, and
   data_ref-combination frequency weights.
2. Schedule: distribute individual synthetic reports across the capture time
   window rather than replaying burst clusters.
3. Generate: for each synthetic report, sample (with fixed seed) a data_ref
   combination from the empirical distributions; advance the per-URCB sequence
   number monotonically; set all fixed fields (service, direction,
   octet_identities, etc.) identically to the real attacks.
4. Output: original rows + synthetic rows, merged and written in timestamp order.
   Synthetic rows carry tag='attack' and frame numbers beyond the real maximum.

Usage:
    python src/augmentation/mms_attack_augmentor.py \\
        --input  data/raw/mms_capture_attack_tags.csv \\
        --output data/raw/mms_capture_attack_tags_augmented.csv.gz \\
        --ratio  0.40 \\
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(10 ** 8)

# ── Schema constants ──────────────────────────────────────────────────────────

ATTACK_TAG = "attack"
NORMAL_TAG = "normal"

# Fixed values shared by all attack rows (observed invariants)
ATK_DIRECTION = "RESPONSE"
ATK_SERVICE = "UNCONFIRMED"
ATK_VLN = "RPT"                  # variable_list_name
ATK_OCTET = "IEDEXPLORER"        # octet_identities (the unauthorized identity)
MIN_SYNTHETIC_ATTACK_GAP_SECONDS = 61.0

# Per-URCB envelope: (urcb_prefix, src_ip, dst_ip, src_port, dst_port, stream_id)
URCB_ENVELOPE: Dict[str, dict] = {
    "E03A103_CTRL/LLN0$RP$A_URCB_10": {
        "src_ip":    "10.0.19.49",
        "src_port":  "102",
        "dst_ip":    "10.0.19.39",
        "dst_port":  "56638",
        "stream_id": "10.0.19.39:56638-10.0.19.49:102",
        "entry_id":  "E03A103_CTRL/LLN0$08002744E921ST0",
    },
    "E01A103_CTRL/LLN0$RP$A_URCB_10": {
        "src_ip":    "10.0.19.47",
        "src_port":  "102",
        "dst_ip":    "10.0.19.39",
        "dst_port":  "56633",
        "stream_id": "10.0.19.39:56633-10.0.19.47:102",
        "entry_id":  "E01A103_CTRL/LLN0$08002744E921ST0",
    },
}

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AttackRow:
    timestamp: datetime
    urcb_ref: str
    seq_num: int
    data_refs: Tuple[str, ...]    # sorted tuple of data references (excl. entry_id)
    raw: dict                     # original CSV dict


@dataclass
class BurstProfile:
    """Statistical model for a single URCB stream, learned from real attacks."""
    urcb_ref: str
    # Burst size distribution (list of observed sizes)
    burst_sizes: List[int] = field(default_factory=list)
    # Intra-burst inter-arrival deltas in seconds
    intra_deltas: List[float] = field(default_factory=list)
    # data_ref combination frequencies  {combo: count}
    combo_counts: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    # Max seq_num seen in real data (synthetic will continue from here)
    max_seq: int = 0
    # Inter-burst gaps in seconds (between last packet of burst n and first of burst n+1)
    inter_gaps: List[float] = field(default_factory=list)
    # Burst start timestamps (for calibrating the time window)
    burst_starts: List[datetime] = field(default_factory=list)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def _ts(s: str) -> datetime:
    try:
        return datetime.strptime(s, TS_FMT)
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def _ts_str(dt: datetime) -> str:
    return dt.strftime(TS_FMT)


# ── Step 1: Profile real attacks ──────────────────────────────────────────────

def profile(rows: List[dict]) -> Dict[str, BurstProfile]:
    """Build per-URCB burst profiles from the 105 real attack rows."""
    attack_rows: List[AttackRow] = []
    for row in rows:
        if row.get("tag") != ATTACK_TAG:
            continue
        ar = json.loads(row["access_result"])
        urcb_ref = ar[0]["value"]
        seq_num = next(e["value"] for e in ar if e["type"] == "unsigned")
        data_refs = tuple(
            sorted(e["value"] for e in ar[2:] if e["type"] == "visible-string")
        )
        attack_rows.append(AttackRow(
            timestamp=_ts(row["timestamp"]),
            urcb_ref=urcb_ref,
            seq_num=seq_num,
            data_refs=data_refs,
            raw=row,
        ))

    attack_rows.sort(key=lambda r: r.timestamp)

    profiles: Dict[str, BurstProfile] = {}
    for urcb_ref in URCB_ENVELOPE:
        profiles[urcb_ref] = BurstProfile(urcb_ref=urcb_ref)

    for urcb_ref, prof in profiles.items():
        stream_rows = [r for r in attack_rows if r.urcb_ref == urcb_ref]
        if not stream_rows:
            continue

        # Segment into bursts (gap > 60 s = new burst)
        bursts: List[List[AttackRow]] = [[stream_rows[0]]]
        for prev, cur in zip(stream_rows, stream_rows[1:]):
            gap = (cur.timestamp - prev.timestamp).total_seconds()
            if gap > 60:
                prof.inter_gaps.append(gap)
                bursts.append([cur])
            else:
                bursts[-1].append(cur)

        prof.burst_starts = [b[0].timestamp for b in bursts]

        for burst in bursts:
            prof.burst_sizes.append(len(burst))
            for a, b in zip(burst, burst[1:]):
                prof.intra_deltas.append((b.timestamp - a.timestamp).total_seconds())
            for r in burst:
                prof.combo_counts[r.data_refs] = prof.combo_counts.get(r.data_refs, 0) + 1

        prof.max_seq = max(r.seq_num for r in stream_rows)

    return profiles


# ── Step 2: Schedule synthetic attack timestamps ──────────────────────────────

def _weighted_choice(rng: random.Random, weights: Dict, total: Optional[int] = None):
    """Weighted random choice from a dict {item: count}."""
    items = list(weights.items())
    population = [item for item, w in items for _ in range(w)]
    return rng.choice(population)


def schedule_bursts(
    profiles: Dict[str, BurstProfile],
    n_synthetic: int,
    window_start: datetime,
    window_end: datetime,
    rng: random.Random,
) -> List[Tuple[str, datetime]]:
    """
    Return one scheduled timestamp per synthetic attack packet.

    The real attack capture contains clustered reports, but the augmented corpus
    should not create artificial IEDEXPLORER bursts. We keep the per-URCB
    proportions and report contents, then spread individual reports across the
    available window.
    """
    # How many packets per URCB, proportional to real counts
    total_real = sum(
        sum(p.burst_sizes) for p in profiles.values() if p.burst_sizes
    )
    scheduled: List[Tuple[str, datetime]] = []

    for urcb_ref, prof in profiles.items():
        if not prof.burst_sizes:
            continue
        real_n = sum(prof.burst_sizes)
        synth_n = round(n_synthetic * real_n / total_real)
        if synth_n == 0:
            continue

        window_seconds = (window_end - window_start).total_seconds()
        slot = max(MIN_SYNTHETIC_ATTACK_GAP_SECONDS, window_seconds / (synth_n + 1))
        for i in range(synth_n):
            t = window_start + timedelta(seconds=slot * (i + 1))
            scheduled.append((urcb_ref, t))

    return scheduled


# ── Step 3: Generate synthetic rows ──────────────────────────────────────────

def generate_synthetic_rows(
    profiles: Dict[str, BurstProfile],
    n_synthetic: int,
    window_start: datetime,
    window_end: datetime,
    base_frame: int,
    rng: random.Random,
) -> List[dict]:
    """Generate n_synthetic attack rows as CSV-ready dicts."""

    burst_schedule = schedule_bursts(
        profiles, n_synthetic, window_start, window_end, rng
    )
    burst_schedule.sort(key=lambda x: x[1])

    # Per-URCB seq counter (continue from real max)
    seq_counters = {u: p.max_seq + 1 for u, p in profiles.items()}

    rows: List[dict] = []
    frame_counter = base_frame

    for urcb_ref, attack_ts in burst_schedule:
        prof = profiles[urcb_ref]
        env = URCB_ENVELOPE[urcb_ref]

        combo = _weighted_choice(rng, prof.combo_counts)

        # Entry ID is always the first element of the observed combo
        entry_id = env["entry_id"]
        data_refs = [r for r in combo if r != entry_id]

        seq = seq_counters[urcb_ref]
        seq_counters[urcb_ref] += 1

        # Build access_result identical in structure to real attacks
        ar_entries = [
            {"type": "visible-string", "value": urcb_ref},
            {"type": "unsigned", "value": seq},
            {"type": "visible-string", "value": entry_id},
        ]
        for ref in data_refs:
            ar_entries.append({"type": "visible-string", "value": ref})

        row = {
            "frame_number":       str(frame_counter),
            "timestamp":          _ts_str(attack_ts),
            "tag":                ATTACK_TAG,
            "direction":          ATK_DIRECTION,
            "service":            ATK_SERVICE,
            "src_ip":             env["src_ip"],
            "src_port":           env["src_port"],
            "dst_ip":             env["dst_ip"],
            "dst_port":           env["dst_port"],
            "stream_id":          env["stream_id"],
            "invoke_id":          "",
            "variable_list_name": ATK_VLN,
            "variables":          "",
            "control_object":     "",
            "control_action":     "",
            "control_value":      "",
            "ctl_num":            "",
            "origin_identifier":  "",
            "origin_category":    "",
            "octet_identities":   ATK_OCTET,
            "access_result":      json.dumps(ar_entries, separators=(",", ":")),
        }
        rows.append(row)
        frame_counter += 1

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  default="data/raw/mms_capture_attack_tags.csv",
                   help="Labelled input CSV (with tag column)")
    p.add_argument("--output", default="data/raw/mms_capture_attack_tags_augmented.csv.gz",
                   help="Output path (.csv or .csv.gz)")
    p.add_argument("--ratio", type=float, default=0.40,
                   help="Target attack fraction (0.40 = 40/60 split, default 0.40)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--profile-only", action="store_true",
                   help="Print feature profiles and exit without writing output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # ── Load input ────────────────────────────────────────────────────────────
    print(f"[1/4] Reading {args.input} ...")
    with _open(args.input) as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        all_rows = list(reader)

    n_total = len(all_rows)
    n_attack = sum(1 for r in all_rows if r.get("tag") == ATTACK_TAG)
    n_normal = n_total - n_attack
    print(f"      {n_total:,} rows  ({n_attack} attack / {n_normal:,} normal)")

    # ── Profile ───────────────────────────────────────────────────────────────
    print("[2/4] Profiling attack patterns ...")
    profiles = profile(all_rows)

    for urcb, prof in profiles.items():
        print(f"\n  URCB: {urcb}")
        if not prof.burst_sizes:
            print("    (no attacks on this stream)")
            continue
        print(f"    Bursts         : {len(prof.burst_sizes)}")
        print(f"    Burst sizes    : min={min(prof.burst_sizes)} max={max(prof.burst_sizes)} "
              f"avg={sum(prof.burst_sizes)/len(prof.burst_sizes):.1f}")
        if prof.intra_deltas:
            print(f"    Intra-burst dt : min={min(prof.intra_deltas):.2f}s "
                  f"max={max(prof.intra_deltas):.2f}s "
                  f"avg={sum(prof.intra_deltas)/len(prof.intra_deltas):.2f}s")
        if prof.inter_gaps:
            print(f"    Inter-burst gap: min={min(prof.inter_gaps):.0f}s "
                  f"max={max(prof.inter_gaps):.0f}s "
                  f"avg={sum(prof.inter_gaps)/len(prof.inter_gaps):.0f}s")
        print(f"    Max seq_num    : {prof.max_seq}")
        print(f"    Data_ref combos: {len(prof.combo_counts)} unique")
        print("    Top combos (F10):")
        for combo, cnt in sorted(prof.combo_counts.items(), key=lambda x: -x[1])[:4]:
            refs = [r.split("/")[-1] for r in combo if "CTRL/LLN0$08" not in r]
            print(f"      {cnt:2d}x  {refs}")

    if args.profile_only:
        return

    # ── Compute target count ──────────────────────────────────────────────────
    # ratio = n_attack_total / (n_attack_total + n_normal)
    # => n_attack_total = ratio * n_normal / (1 - ratio)
    target_attack = math.ceil(args.ratio * n_normal / (1.0 - args.ratio))
    n_synthetic = max(0, target_attack - n_attack)
    print(f"\n[3/4] Generating {n_synthetic:,} synthetic attack rows "
          f"(target ratio {args.ratio:.0%}: {target_attack:,} attacks / {n_normal:,} normal)")

    if n_synthetic == 0:
        print("      Nothing to do — dataset already meets or exceeds the target ratio.")
    else:
        # Time window: span of real attack timestamps (same as capture window)
        attack_rows_raw = [r for r in all_rows if r.get("tag") == ATTACK_TAG]
        all_ts = [_ts(r["timestamp"]) for r in all_rows if r.get("timestamp")]
        window_start = min(all_ts)
        window_end   = max(all_ts)
        base_frame = max(int(r.get("frame_number", 0) or 0) for r in all_rows) + 1

        synthetic = generate_synthetic_rows(
            profiles, n_synthetic, window_start, window_end, base_frame, rng
        )
        print(f"      Generated {len(synthetic):,} rows")

        all_rows.extend(synthetic)
        # Sort by timestamp so the output reads chronologically
        all_rows.sort(key=lambda r: r.get("timestamp", ""))

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"[4/4] Writing {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with _open(args.output, "wt") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    n_out_attack = sum(1 for r in all_rows if r.get("tag") == ATTACK_TAG)
    n_out_total  = len(all_rows)
    actual_ratio = n_out_attack / n_out_total
    print(f"      {n_out_total:,} rows total  "
          f"({n_out_attack:,} attack / {n_out_total - n_out_attack:,} normal)  "
          f"ratio={actual_ratio:.3f}")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
