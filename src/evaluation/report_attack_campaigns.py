"""
Attack Campaign Report
Groups the 105 attack packets into distinct sessions based on time gaps,
revealing how many separate attack campaigns the attacker ran.
Output: results/report_attack_campaigns.csv and results/report_attack_campaigns.txt
"""
import pandas as pd

EVENTS_FILE   = "results/ids_attack_events.csv"
OUT_CSV       = "results/report_attack_campaigns.csv"
OUT_TXT       = "results/report_attack_campaigns.txt"
SESSION_GAP_M = 30  # minutes of silence = new campaign

df = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])
packets = df.drop_duplicates(subset="line_number").copy()
packets = packets.sort_values("timestamp").reset_index(drop=True)

# Assign campaign IDs based on time gap
packets["gap_min"] = packets["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
packets["campaign"] = (packets["gap_min"] > SESSION_GAP_M).cumsum() + 1

# Build per-campaign summary including all object types seen
events_full = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])
events_full = events_full.merge(
    packets[["line_number", "campaign", "gap_min"]],
    on="line_number", how="left"
)

campaigns = []
for cid, grp in events_full.groupby("campaign"):
    pkt_lines = grp["line_number"].unique()
    start     = grp["timestamp"].min()
    end       = grp["timestamp"].max()
    duration  = (end - start).total_seconds()
    types     = sorted(grp["switchgear"].unique())
    campaigns.append({
        "campaign":       cid,
        "start":          start,
        "end":            end,
        "duration_sec":   int(duration),
        "attack_packets": len(pkt_lines),
        "switchgear_types": ", ".join(types),
    })

camp_df = pd.DataFrame(campaigns)
camp_df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("=" * 70)
lines.append("  ATTACK CAMPAIGN REPORT")
lines.append(f"  (session gap threshold: {SESSION_GAP_M} minutes)")
lines.append("=" * 70)
lines.append(f"\nTotal distinct attack campaigns: {len(camp_df)}")
lines.append(f"Total attack packets across all campaigns: {camp_df['attack_packets'].sum()}\n")

for _, row in camp_df.iterrows():
    dur = row["duration_sec"]
    dur_str = f"{dur // 60}m {dur % 60}s" if dur >= 60 else f"{dur}s"
    lines.append(f"Campaign {int(row['campaign'])}")
    lines.append(f"  Start:    {row['start']}")
    lines.append(f"  End:      {row['end']}")
    lines.append(f"  Duration: {dur_str}")
    lines.append(f"  Packets:  {int(row['attack_packets'])}")
    lines.append(f"  Targets:  {row['switchgear_types']}")
    lines.append("")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, "w") as f:
    f.write(report + "\n")

print(f"Saved:\n  {OUT_CSV}\n  {OUT_TXT}")
