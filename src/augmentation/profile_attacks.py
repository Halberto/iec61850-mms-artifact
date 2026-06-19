"""Study stage: profile the real attack packets in the normalized corpus.

Reads the captured corpus, isolates the real attack rows (``tag == attack``),
and learns the statistical structure the generative model reproduces:

  * constant fingerprint fields (always identical on attack packets)
  * categorical PMFs (src_ip, IED, members-per-packet, ...)
  * per-IED controllable-object vocabulary, split by CSWI / CILO so the
    object-type <-> status-suffix relationship stays valid
  * a first-order Markov chain over the object-token sequence within a packet
    (per IED + object type) to reproduce realistic report ordering
  * per-stream report-sequence start + increment distribution
  * per-stream inter-arrival timing

Outputs:
  results/attack_profile.json      -- machine-readable model parameters
  results/attack_templates.jsonl   -- the real attack rows used as scaffolds
  results/attack_profile_report.md -- human-readable summary (the "study")

This stage is read-only with respect to the corpus.
"""

import argparse
import csv
import gzip
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime

csv.field_size_limit(10**8)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fields that are byte-for-byte constant across attack packets are reproduced
# deterministically rather than sampled.
FINGERPRINT_FIELDS = [
    "tag",
    "direction",
    "service",
    "variable_list_name",
    "octet_identities",
    "dst_ip",
    "origin_category",
]

OBJECT_RE = re.compile(r"([A-Z0-9]+_CTRL)/(Q\dCSWI\d|Q\dCILO\d)")
IED_RE = re.compile(r"([A-Z0-9]+_CTRL)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default=os.path.join(REPO_ROOT, "data", "raw", "mms_capture_normalized.csv.gz"),
        help="Normalized corpus (.csv or .csv.gz).",
    )
    parser.add_argument("--tag-column", default="tag")
    parser.add_argument("--attack-value", default="attack")
    parser.add_argument(
        "--profile-json",
        default=os.path.join(REPO_ROOT, "results", "attack_profile.json"),
    )
    parser.add_argument(
        "--templates-jsonl",
        default=os.path.join(REPO_ROOT, "results", "attack_templates.jsonl"),
    )
    parser.add_argument(
        "--report-md",
        default=os.path.join(REPO_ROOT, "results", "attack_profile_report.md"),
    )
    return parser.parse_args()


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def object_type(token: str) -> str:
    return "CSWI" if "CSWI" in token else "CILO"


def parse_timestamp(value: str):
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def object_token_sequence(access_raw: str) -> list[tuple[str, str]]:
    """Ordered (IED, token) pairs as they appear in a packet's access_result."""
    return [(m.group(1), m.group(2)) for m in OBJECT_RE.finditer(access_raw or "")]


def increments(sorted_values: list[int]) -> list[int]:
    return [b - a for a, b in zip(sorted_values, sorted_values[1:]) if b - a >= 0]


def build_profile(attack_rows: list[dict]) -> dict:
    # --- constants / fingerprint ---
    constants: dict[str, str] = {}
    for field in FINGERPRINT_FIELDS:
        values = {(r.get(field) or "").strip() for r in attack_rows}
        constants[field] = next(iter(values)) if len(values) == 1 else ""

    # --- categorical PMFs ---
    src_ip_pmf = Counter((r.get("src_ip") or "").strip() for r in attack_rows)
    ied_pmf: Counter = Counter()
    members_pmf: Counter = Counter()

    ied_urcb: dict[str, Counter] = defaultdict(Counter)
    ied_objects: dict[str, dict[str, Counter]] = defaultdict(lambda: {"CSWI": Counter(), "CILO": Counter()})
    # Markov transitions over object tokens within a packet, per IED + type.
    markov: dict[str, dict[str, dict[str, Counter]]] = defaultdict(
        lambda: {"CSWI": defaultdict(Counter), "CILO": defaultdict(Counter)}
    )
    markov_start: dict[str, dict[str, Counter]] = defaultdict(lambda: {"CSWI": Counter(), "CILO": Counter()})

    # --- per-stream sequence + timing ---
    stream_seq: dict[str, list[int]] = defaultdict(list)
    stream_times: dict[str, list[datetime]] = defaultdict(list)
    # channel structural fields (inherited verbatim by synthetic rows)
    channel_fields: dict[str, dict] = {}

    for row in attack_rows:
        access_raw = row.get("access_result") or ""
        stream = (row.get("stream_id") or "").strip()
        primary_ied = IED_RE.search(access_raw)
        primary_ied = primary_ied.group(1) if primary_ied else ""
        if primary_ied:
            ied_pmf[primary_ied] += 1

        for block in re.findall(r"[A-Z0-9]+_CTRL/LLN0[$]RP[$]\w+", access_raw):
            ied_urcb[block.split("/", 1)[0]][block] += 1

        tokens = object_token_sequence(access_raw)
        members_pmf[len(tokens)] += 1
        per_type_seq: dict[str, list[str]] = {"CSWI": [], "CILO": []}
        for ied, token in tokens:
            t = object_type(token)
            ied_objects[ied][t][token] += 1
            per_type_seq[t].append((ied, token))
        for t, seq in per_type_seq.items():
            if not seq:
                continue
            ied0, tok0 = seq[0]
            markov_start[ied0][t][tok0] += 1
            for (ied_a, tok_a), (ied_b, tok_b) in zip(seq, seq[1:]):
                markov[ied_a][t][tok_a][tok_b] += 1

        try:
            items = json.loads(access_raw)
            seqval = next(
                (it["value"] for it in items if isinstance(it, dict) and it.get("type") == "unsigned"),
                None,
            )
            if isinstance(seqval, int):
                stream_seq[stream].append(seqval)
        except (json.JSONDecodeError, TypeError):
            pass

        ts = parse_timestamp((row.get("timestamp") or "").strip())
        if ts:
            stream_times[stream].append(ts)

        channel_fields.setdefault(
            (row.get("src_ip") or "").strip(),
            {
                "src_ip": (row.get("src_ip") or "").strip(),
                "src_port": row.get("src_port") or "",
                "dst_ip": (row.get("dst_ip") or "").strip(),
                "dst_port": row.get("dst_port") or "",
                "stream_id": stream,
                "src_mac": row.get("src_mac") or "",
                "dst_mac": row.get("dst_mac") or "",
                "ipv4_ttl": row.get("ipv4_ttl") or "",
                "tcp_flags": row.get("tcp_flags") or "",
                "tcp_window": row.get("tcp_window") or "",
            },
        )

    # --- summarize per-stream sequence increments + timing ---
    stream_seq_model = {}
    for stream, values in stream_seq.items():
        ordered = sorted(values)
        incs = increments(ordered) or [1]
        stream_seq_model[stream] = {
            "start_min": int(ordered[0]),
            "start_max": int(ordered[-1]),
            "increment_pmf": dict(Counter(incs)),
        }

    stream_timing_model = {}
    for stream, times in stream_times.items():
        ordered = sorted(times)
        gaps = [round((b - a).total_seconds(), 3) for a, b in zip(ordered, ordered[1:])]
        gaps = [g for g in gaps if g >= 0] or [17.0]
        stream_timing_model[stream] = {
            "start_timestamp": ordered[0].isoformat(),
            "gap_seconds_sample": gaps,
            "gap_median": statistics.median(gaps),
            "gap_mean": round(statistics.mean(gaps), 3),
        }

    def nested_counter_to_dict(obj):
        if isinstance(obj, Counter):
            return {str(k): int(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: nested_counter_to_dict(v) for k, v in obj.items()}
        return obj

    profile = {
        "n_attacks": len(attack_rows),
        "constants": constants,
        "src_ip_pmf": dict(src_ip_pmf),
        "ied_pmf": dict(ied_pmf),
        "members_pmf": {str(k): int(v) for k, v in members_pmf.items()},
        "ied_urcb": {ied: dict(c) for ied, c in ied_urcb.items()},
        "ied_objects": {
            ied: {t: dict(c) for t, c in types.items()} for ied, types in ied_objects.items()
        },
        "object_markov_start": nested_counter_to_dict(markov_start),
        "object_markov": nested_counter_to_dict(markov),
        "stream_seq": stream_seq_model,
        "stream_timing": stream_timing_model,
        "channel_fields": channel_fields,
    }
    return profile


def write_report(profile: dict, path: str) -> None:
    objects = profile["ied_objects"]
    n_objects = sum(len(types["CSWI"]) + len(types["CILO"]) for types in objects.values())
    lines = [
        "# Attack-packet study (generative-model inputs)",
        "",
        f"Real attack packets profiled: **{profile['n_attacks']}**",
        "",
        "## Constant fingerprint fields",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for field, value in profile["constants"].items():
        lines.append(f"| `{field}` | `{value or '(empty)'}` |")
    lines += [
        "",
        "## Categorical distributions",
        "",
        f"- `src_ip`: {profile['src_ip_pmf']}",
        f"- IED: {profile['ied_pmf']}",
        f"- report members per packet: {profile['members_pmf']}",
        f"- distinct controllable objects: **{n_objects}** across {len(objects)} IEDs",
        "",
        "### Per-IED controllable objects",
        "",
    ]
    for ied, types in objects.items():
        lines.append(f"- **{ied}** — CSWI: {types['CSWI']} · CILO: {types['CILO']}")
    lines += ["", "## Per-stream report sequence", ""]
    for stream, model in profile["stream_seq"].items():
        lines.append(
            f"- `{stream}`: start {model['start_min']}..{model['start_max']}, "
            f"increment PMF {model['increment_pmf']}"
        )
    lines += ["", "## Per-stream timing", ""]
    for stream, model in profile["stream_timing"].items():
        lines.append(
            f"- `{stream}`: median gap {model['gap_median']}s, mean {model['gap_mean']}s, "
            f"n_gaps {len(model['gap_seconds_sample'])}"
        )
    lines += [
        "",
        "## Model note",
        "",
        "Synthetic attacks are produced by `src/augmentation/attack_model.py`, which samples "
        "these distributions and applies them onto real attack scaffolds (preserving protocol "
        "structure and the CSWI/`$ST$Pos` vs CILO/`$ST$EnaCls` relationship). See "
        "`docs/dataset_description.md`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    print(f"Reading corpus: {args.corpus}")
    attack_rows: list[dict] = []
    total = 0
    with open_maybe_gzip(args.corpus) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            total += 1
            if (row.get(args.tag_column) or "").strip().lower() == args.attack_value.lower():
                attack_rows.append(row)
    print(f"Corpus rows: {total} | attack rows: {len(attack_rows)}")
    if not attack_rows:
        raise ValueError("No attack rows found; check --tag-column / --attack-value.")

    profile = build_profile(attack_rows)
    profile["corpus_fields"] = fieldnames
    profile["corpus_total_rows"] = total

    os.makedirs(os.path.dirname(args.profile_json), exist_ok=True)
    with open(args.profile_json, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    print(f"Wrote profile -> {args.profile_json}")

    with open(args.templates_jsonl, "w", encoding="utf-8") as handle:
        for row in attack_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(attack_rows)} attack templates -> {args.templates_jsonl}")

    write_report(profile, args.report_md)
    print(f"Wrote study report -> {args.report_md}")


if __name__ == "__main__":
    main()
