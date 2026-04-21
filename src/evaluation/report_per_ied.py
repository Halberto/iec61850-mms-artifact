"""Build a per-IED attack-packet summary from IDS attack events."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_FILE = BASE_DIR / "results" / "ids_attack_events.csv"
OUT_CSV = BASE_DIR / "results" / "report_per_ied.csv"
OUT_TXT = BASE_DIR / "results" / "report_per_ied.txt"


def main() -> None:
    df = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])
    df["ied"] = df["data_object"].str.extract(r"^([^_]+)")[0]

    ied_summary = df.groupby(["ied", "switchgear"])["line_number"].nunique().reset_index()
    ied_summary.columns = ["ied", "switchgear", "attack_packets"]
    ied_summary = ied_summary.sort_values(["ied", "switchgear"]).reset_index(drop=True)

    ied_totals = ied_summary.groupby("ied")["attack_packets"].sum().reset_index()
    ied_totals.columns = ["ied", "total_attack_packets"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ied_summary.to_csv(OUT_CSV, index=False)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  PER-IED BREAKDOWN REPORT")
    lines.append("=" * 60)
    lines.append("")

    for ied, group in ied_summary.groupby("ied"):
        total = ied_totals.loc[ied_totals["ied"] == ied, "total_attack_packets"].values[0]
        lines.append(f"IED: {ied}  (total attack packets: {total})")
        lines.append(f"  {'Switchgear Type':<20} {'Attack Packets':>15}")
        lines.append(f"  {'-' * 20} {'-' * 15}")
        for _, row in group.iterrows():
            lines.append(f"  {row['switchgear']:<20} {int(row['attack_packets']):>15}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("NOTE: One attack packet can target objects on multiple IEDs,")
    lines.append("      so totals may exceed 105 unique attack packets.")

    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved:\n  {OUT_CSV}\n  {OUT_TXT}")


if __name__ == "__main__":
    main()
