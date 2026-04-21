"""
Temporal Analysis Report
Analyses the distribution of attack packets across the 56-hour capture window.
Output: results/report_temporal_analysis.csv and results/report_temporal_analysis.txt
"""
import pandas as pd

EVENTS_FILE = "results/ids_attack_events.csv"
OUT_CSV     = "results/report_temporal_analysis.csv"
OUT_TXT     = "results/report_temporal_analysis.txt"

df = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])
packets = df.drop_duplicates(subset="line_number").copy()
packets = packets.sort_values("timestamp").reset_index(drop=True)

# Hourly distribution
packets["hour"] = packets["timestamp"].dt.floor("h")
hourly = packets.groupby("hour").size().reset_index(name="attack_packets")
hourly["hour_label"] = hourly["hour"].dt.strftime("%Y-%m-%d %H:00")

# Daily distribution
packets["date"] = packets["timestamp"].dt.date
daily = packets.groupby("date").size().reset_index(name="attack_packets")

# Time span
capture_start = packets["timestamp"].min()
capture_end   = packets["timestamp"].max()
duration_h    = (capture_end - capture_start).total_seconds() / 3600

hourly[["hour_label", "attack_packets"]].to_csv(OUT_CSV, index=False)

lines = []
lines.append("=" * 60)
lines.append("  TEMPORAL ANALYSIS REPORT")
lines.append("=" * 60)
lines.append(f"\nAttack window: {capture_start} --> {capture_end}")
lines.append(f"Span: {duration_h:.1f} hours across {packets['date'].nunique()} days\n")

lines.append(f"Total unique attack packets: {len(packets)}")
lines.append(f"Hourly buckets with attacks: {len(hourly)}\n")

lines.append("-" * 40)
lines.append(f"{'Hour':<22} {'Packets':>8}")
lines.append("-" * 40)
for _, row in hourly.iterrows():
    lines.append(f"{row['hour_label']:<22} {int(row['attack_packets']):>8}")

lines.append("\n" + "-" * 40)
lines.append(f"{'Date':<12} {'Packets':>8}")
lines.append("-" * 40)
for _, row in daily.iterrows():
    lines.append(f"{str(row['date']):<12} {int(row['attack_packets']):>8}")

lines.append("\n" + "=" * 60)
lines.append(f"Peak hour: {hourly.loc[hourly['attack_packets'].idxmax(), 'hour_label']} "
             f"({hourly['attack_packets'].max()} packets)")
lines.append(f"Quietest attack hour: {hourly.loc[hourly['attack_packets'].idxmin(), 'hour_label']} "
             f"({hourly['attack_packets'].min()} packets)")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, "w") as f:
    f.write(report + "\n")

print(f"\nSaved:\n  {OUT_CSV}\n  {OUT_TXT}")
