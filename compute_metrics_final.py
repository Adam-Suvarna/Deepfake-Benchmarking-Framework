import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\adams\OneDrive - Coventry University\Disseertation Project\Image Dataset\The Evauluation Dataset"

RESULT_FILES = {
    "M1_MesoNet":            "results_mesonet.csv",
    "M2_XceptionNet":        "results_xceptionnet.csv",
    "M3_SBI_EfficientNetB4": "results_sbi.csv",
    "M4_EfficientNetB4":     "results_efficientnet_b4.csv",
    "M5_EfficientNetB0":     "results_efficientnet_b0.csv",
}

ROBUSTNESS_FILES = {
    "M1_MesoNet":            "results_mesonet_robustness.csv",
    "M2_XceptionNet":        "results_xceptionnet_robustness.csv",
    "M3_SBI_EfficientNetB4": "results_sbi_robustness.csv",
    "M4_EfficientNetB4":     "results_efficientnet_b4_robustness.csv",
    "M5_EfficientNetB0":     "results_efficientnet_b0_robustness.csv",
}

OUTPUT_CSV = os.path.join(DATASET_DIR, "final_ranking.csv")

# ── Scoring weights ───────────────────────────────────────────────────────────
W_PERF   = 0.40
W_GEN    = 0.25
W_ROBUST = 0.25
W_EFF    = 0.10

# ── Bootstrap confidence interval ────────────────────────────────────────────
def bootstrap_ci(y_true, y_scores, metric_fn, n=1000, ci=0.95):
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt = np.array(y_true)[idx]
        ys = np.array(y_scores)[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, ys))
        except:
            continue
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return lower, upper

# ── Runtime score (lower is better, normalised to 0-1) ───────────────────────
def runtime_score(median_rt, max_rt=0.20):
    return round(1.0 - min(median_rt / max_rt, 1.0), 4)

# ── Robustness AUC across all conditions ─────────────────────────────────────
def compute_robustness_score(rob_df):
    conditions = rob_df["condition"].unique()
    auc_scores = []
    for cond in conditions:
        sub = rob_df[rob_df["condition"] == cond]
        yt = [1 if l == "FAKE" else 0 for l in sub["true_label"]]
        ys = sub["fake_score"].tolist()
        if len(np.unique(yt)) < 2:
            continue
        try:
            auc_scores.append(roc_auc_score(yt, ys))
        except:
            continue
    return round(np.mean(auc_scores), 4) if auc_scores else 0.5

# ── Process each detector ─────────────────────────────────────────────────────
results_summary = []

for detector_name, filename in RESULT_FILES.items():
    filepath = os.path.join(DATASET_DIR, filename)
    rob_filepath = os.path.join(DATASET_DIR, ROBUSTNESS_FILES[detector_name])
    print(f"\nProcessing: {detector_name}")

    df     = pd.read_csv(filepath)
    rob_df = pd.read_csv(rob_filepath)

    # Binary labels
    y_true  = [1 if l == "FAKE" else 0 for l in df["true_label"]]
    y_pred  = [1 if l == "FAKE" else 0 for l in df["predicted_label"]]
    y_score = df["fake_score"].tolist()

    # ── Overall metrics ───────────────────────────────────────────────────────
    auc = roc_auc_score(y_true, y_score)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Bootstrap CIs
    auc_low, auc_high = bootstrap_ci(y_true, y_score, roc_auc_score)
    f1_low,  f1_high  = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, zero_division=0)
    )

    # ── Generalisation (Fake-A vs Fake-B) ────────────────────────────────────
    df_a = df[df["generator"].isin(["A", "REAL"])]
    df_b = df[df["generator"].isin(["B", "REAL"])]

    def sub_auc(sub_df):
        yt = [1 if l == "FAKE" else 0 for l in sub_df["true_label"]]
        ys = sub_df["fake_score"].tolist()
        if len(np.unique(yt)) < 2:
            return 0.5
        return roc_auc_score(yt, ys)

    auc_a   = sub_auc(df_a)
    auc_b   = sub_auc(df_b)
    gen_avg = (auc_a + auc_b) / 2
    auc_gap = abs(auc_a - auc_b)

    # ── Robustness (real scores from robustness CSV) ──────────────────────────
    robust_score = compute_robustness_score(rob_df)

    # ── Runtime ───────────────────────────────────────────────────────────────
    median_rt = df["runtime_sec"].median()
    eff_score = runtime_score(median_rt)

    # ── Performance score ─────────────────────────────────────────────────────
    perf_score = (auc + f1) / 2

    # ── Final weighted score ──────────────────────────────────────────────────
    total_pct = (W_PERF   * perf_score  +
                 W_GEN    * gen_avg     +
                 W_ROBUST * robust_score +
                 W_EFF    * eff_score)
    final_score = round(total_pct * 10, 2)

    print(f"  AUC:          {auc:.4f} [{auc_low:.4f} - {auc_high:.4f}]")
    print(f"  F1:           {f1:.4f} [{f1_low:.4f} - {f1_high:.4f}]")
    print(f"  TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    print(f"  AUC Fake-A:   {auc_a:.4f}")
    print(f"  AUC Fake-B:   {auc_b:.4f}")
    print(f"  AUC Gap:      {auc_gap:.4f}")
    print(f"  Median RT:    {median_rt:.4f}s")
    print(f"  Perf Score:   {perf_score:.4f}")
    print(f"  Gen Score:    {gen_avg:.4f}")
    print(f"  Robust Score: {robust_score:.4f}")
    print(f"  Eff Score:    {eff_score:.4f}")
    print(f"  FINAL SCORE:  {final_score}/10")

    results_summary.append({
        "detector":       detector_name,
        "auc":            round(auc, 4),
        "auc_ci_low":     round(auc_low, 4),
        "auc_ci_high":    round(auc_high, 4),
        "f1":             round(f1, 4),
        "f1_ci_low":      round(f1_low, 4),
        "f1_ci_high":     round(f1_high, 4),
        "tp":             int(tp),
        "tn":             int(tn),
        "fp":             int(fp),
        "fn":             int(fn),
        "auc_fakeA":      round(auc_a, 4),
        "auc_fakeB":      round(auc_b, 4),
        "auc_gap":        round(auc_gap, 4),
        "gen_avg":        round(gen_avg, 4),
        "median_rt_sec":  round(median_rt, 4),
        "perf_score":     round(perf_score, 4),
        "robust_score":   round(robust_score, 4),
        "eff_score":      round(eff_score, 4),
        "final_score":    final_score
    })

# ── Sort by final score ───────────────────────────────────────────────────────
results_summary.sort(key=lambda x: x["final_score"], reverse=True)

# ── Save to CSV ───────────────────────────────────────────────────────────────
fieldnames = list(results_summary[0].keys())
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_summary)

print("\n" + "="*60)
print("FINAL RANKING")
print("="*60)
for i, r in enumerate(results_summary):
    print(f"  #{i+1}  {r['detector']:<30} {r['final_score']}/10")
print("="*60)
print(f"\nFull results saved to: {OUTPUT_CSV}")
