"""Validate that synthetic attacks reproduce the real attack distributions.

Draws a sample from the fitted model and compares its per-field marginals to
the real attack packets with a chi-square goodness-of-fit test (categorical
fields) and a coverage check (no out-of-vocabulary values). Writes a fidelity
report to results/augmentation_fidelity.md.

This is a lightweight, model-only check -- it does not require generating the
full extended corpus.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_model import AttackGenerativeModel  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJECT_RE = re.compile(r"(Q\dCSWI\d|Q\dCILO\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", default=os.path.join(REPO_ROOT, "results", "attack_profile.json"))
    parser.add_argument("--templates-jsonl", default=os.path.join(REPO_ROOT, "results", "attack_templates.jsonl"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--report-md", default=os.path.join(REPO_ROOT, "results", "augmentation_fidelity.md"))
    return parser.parse_args()


def chi_square(observed: Counter, expected_pmf: dict, total: int) -> tuple[float, int]:
    """Pearson chi-square statistic of observed counts vs an expected PMF."""
    exp_total = sum(expected_pmf.values())
    stat = 0.0
    keys = set(observed) | set(expected_pmf)
    for key in keys:
        exp = expected_pmf.get(key, 0) / exp_total * total if exp_total else 0
        obs = observed.get(key, 0)
        if exp > 0:
            stat += (obs - exp) ** 2 / exp
        elif obs > 0:
            stat += obs  # mass where the model expected none
    return stat, max(0, len(keys) - 1)


def object_counts(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for tok in OBJECT_RE.findall(r.get("access_result") or ""):
            c[tok] += 1
    return c


def field_counts(rows: list[dict], field: str) -> Counter:
    return Counter((r.get(field) or "").strip() for r in rows)


def main() -> None:
    args = parse_args()
    model = AttackGenerativeModel.from_files(args.profile_json, args.templates_jsonl)
    real = model.templates
    rng = random.Random(int(model.profile.get("random_seed", 12345)) if isinstance(model.profile.get("random_seed"), int) else 12345)
    synth = list(model.sample(args.sample_size, rng))

    lines = [
        "# Augmentation fidelity report",
        "",
        f"Real attack packets: {len(real)} · synthetic sampled: {len(synth)}",
        "",
        "Chi-square goodness-of-fit of synthetic marginals vs the real-attack PMF "
        "(lower normalized statistic = closer; coverage check flags any value the "
        "model emitted that never occurred in real data).",
        "",
        "| field | real categories | synthetic categories | out-of-vocab | chi2/df |",
        "|---|---|---|---|---|",
    ]

    checks = {
        "src_ip": lambda rows: field_counts(rows, "src_ip"),
        "stream_id": lambda rows: field_counts(rows, "stream_id"),
        "controllable_object": object_counts,
    }
    all_ok = True
    for name, fn in checks.items():
        real_c = fn(real)
        syn_c = fn(synth)
        oov = sorted(set(syn_c) - set(real_c))
        stat, df = chi_square(syn_c, real_c, sum(syn_c.values()))
        norm = stat / df if df else 0.0
        if oov:
            all_ok = False
        lines.append(
            f"| {name} | {len(real_c)} | {len(syn_c)} | {('NONE' if not oov else ', '.join(oov))} | {norm:.3f} |"
        )

    # Fingerprint constants must be identical on every synthetic row.
    fp_fields = ["direction", "service", "variable_list_name", "octet_identities", "dst_ip"]
    fp_ok = all(
        all((r.get(f) or "") == model.constants.get(f, "") for f in fp_fields) for r in synth
    )
    all_ok = all_ok and fp_ok

    lines += [
        "",
        f"- Fingerprint constants identical on all synthetic rows: **{fp_ok}**",
        f"- No out-of-vocabulary values in any checked field: **{all_ok}**",
        "",
        "_Faithful by construction: the model samples only real, observed values and "
        "applies them onto real attack scaffolds._",
        "",
    ]

    os.makedirs(os.path.dirname(args.report_md), exist_ok=True)
    with open(args.report_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Fidelity report -> {args.report_md}")
    print("Overall fidelity OK:", all_ok)


if __name__ == "__main__":
    main()
