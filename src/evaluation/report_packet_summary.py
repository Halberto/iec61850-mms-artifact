"""
Packet-Level Summary Report
One row per unique attack packet (105 rows), listing every object
targeted in that packet — cleaner than ids_attack_events.csv for readers.
Output: results/report_packet_summary.csv and results/report_packet_summary.txt
"""
import pandas as pd

EVENTS_FILE = "results/ids_attack_events.csv"
OUT_CSV     = "results/report_packet_summary.csv"
OUT_TXT     = "results/report_packet_summary.txt"

df = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"])

summary = (
    df.sort_values(["line_number", "switchgear"])
    .groupby(["line_number", "timestamp", "src_ip", "dst_ip", "rogue_origin"])
    .agg(
        objects_targeted=("data_object",  lambda x: "; ".join(sorted(x.unique()))),
        switchgear_types=("switchgear",   lambda x: ", ".join(sorted(x.unique()))),
        n_objects=       ("data_object",  "nunique"),
    )
    .reset_index()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

summary.index += 1
summary.index.name = "seq"
summary.to_csv(OUT_CSV)

lines = []
lines.append("=" * 70)
lines.append("  PACKET-LEVEL SUMMARY REPORT")
lines.append(f"  Total unique attack packets: {len(summary)}")
lines.append("=" * 70)
lines.append("")
lines.append(f"  {'Seq':<4} {'Timestamp':<26} {'Src IP':<14} {'Switchgear Types':<40} {'Objects'}")
lines.append(f"  {'-'*4} {'-'*26} {'-'*14} {'-'*40} {'-'*6}")

for seq, row in summary.iterrows():
    ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"  {seq:<4} {ts:<26} {row['src_ip']:<14} {row['switchgear_types']:<40} {int(row['n_objects'])}")

lines.append("")
lines.append("=" * 70)
lines.append("Switchgear type counts across all packets:")
type_counts = (
    df.groupby("switchgear")["line_number"].nunique()
    .sort_values(ascending=False)
)
for sw, cnt in type_counts.items():
    lines.append(f"  {sw:<20} {cnt} packets")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, "w") as f:
    f.write(report + "\n")

print(f"\nSaved:\n  {OUT_CSV}\n  {OUT_TXT}")
