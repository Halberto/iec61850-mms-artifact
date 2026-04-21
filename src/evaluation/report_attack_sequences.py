"""Build a per-campaign ordered sequence report from IDS attack events."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_FILE = BASE_DIR / "results" / "ids_attack_events.csv"
OUT_CSV = BASE_DIR / "results" / "report_attack_sequences.csv"
OUT_TXT = BASE_DIR / "results" / "report_attack_sequences.txt"
SESSION_GAP_MINUTES = 30


def main() -> None:
    df = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])
    packets = df.drop_duplicates(subset="line_number").copy()
    packets = packets.sort_values("timestamp").reset_index(drop=True)

    packets["gap_min"] = packets["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    packets["campaign"] = (packets["gap_min"] > SESSION_GAP_MINUTES).cumsum() + 1

    df = df.merge(packets[["line_number", "campaign"]], on="line_number", how="left")

    packet_view = (
        df.sort_values(["campaign", "timestamp", "line_number"])
        .groupby(["campaign", "line_number", "timestamp"])
        .agg(switchgear_hit=("switchgear", lambda values: " + ".join(sorted(values.unique()))))
        .reset_index()
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    packet_view.to_csv(OUT_CSV, index=False)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  ATTACK SEQUENCE REPORT")
    lines.append(f"  (session gap threshold: {SESSION_GAP_MINUTES} minutes)")
    lines.append("=" * 70)

    for campaign_id, group in packet_view.groupby("campaign"):
        lines.append(
            f"\nCampaign {campaign_id}  ({len(group)} packets, "
            f"{group['timestamp'].min().strftime('%Y-%m-%d %H:%M')} - "
            f"{group['timestamp'].max().strftime('%H:%M')})"
        )
        lines.append(f"  {'#':<4} {'Timestamp':<30} {'Switchgear Targeted'}")
        lines.append(f"  {'-' * 4} {'-' * 30} {'-' * 35}")
        for index, (_, row) in enumerate(group.iterrows(), 1):
            timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"  {index:<4} {timestamp:<30} {row['switchgear_hit']}")

    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved:\n  {OUT_CSV}\n  {OUT_TXT}")


if __name__ == "__main__":
    main()
