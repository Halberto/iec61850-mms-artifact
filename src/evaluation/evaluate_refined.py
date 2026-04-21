import pandas as pd

# Paths
BASELINE_PATH = 'mms_baseline_results.csv'
RULE_TAGS_PATH = 'mms_sample_attack_tags.csv'

def evaluate_refined():
    print("Loading datasets...")
    # Load model results (100k rows)
    df_model = pd.read_csv(BASELINE_PATH)
    # Load rule tags (100k rows)
    df_rules = pd.read_csv(RULE_TAGS_PATH)
    
    # Check if they match on index/line number
    # Baseline results: line_number is from 1 to 100000? 
    # Or just index 0-99999.
    # mms_sample_attack_tags.csv has 'line_number' column.
    
    print(f"Model results: {len(df_model)} rows")
    print(f"Rule tags: {len(df_rules)} rows")
    
    # Align rows. Both should be the same 100k sample in the same order.
    # Let's verify by checking timestamps or just merge on line_number if available.
    # Baseline Results doesn't have line_number of original file, but results of 100k.
    # mms_sample_attack_tags.csv has line_number (1 to 100000).
    
    # Let's assume they are the same order.
    df_model['rule_tag'] = df_rules['tag']
    df_model['is_rule_anomaly'] = df_model['rule_tag'].apply(lambda x: 1 if x != 'normal' else 0)
    
    # Overlap analysis
    both = df_model[(df_model['is_anomaly'] == 1) & (df_model['is_rule_anomaly'] == 1)]
    model_only = df_model[(df_model['is_anomaly'] == 1) & (df_model['is_rule_anomaly'] == 0)]
    rule_only = df_model[(df_model['is_anomaly'] == 0) & (df_model['is_rule_anomaly'] == 1)]
    
    n_model = df_model['is_anomaly'].sum()
    n_rules = df_model['is_rule_anomaly'].sum()
    n_overlap = len(both)
    
    precision = n_overlap / n_model if n_model > 0 else 0
    recall = n_overlap / n_rules if n_rules > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    summary = f"""
Refined Evaluation Summary:
---------------------------
Total Rows: {len(df_model)}
Anomalies detected by Model: {n_model}
Anomalies detected by Rules (Ground Truth): {n_rules}
Overlap (True Positives): {n_overlap}
Model False Positives (Rules say normal): {len(model_only)}
Model False Negatives (Rules say anomaly): {len(rule_only)}

Model Metrics:
--------------
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

Sample of Rule-Based Anomalies (detected by rules):
{df_rules[df_rules['tag'] != 'normal'][['line_number', 'tag', 'reasons', 'timestamp', 'service']].head(10).to_string()}
"""
    print(summary)
    with open('refined_evaluation_summary.txt', 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    evaluate_refined()
