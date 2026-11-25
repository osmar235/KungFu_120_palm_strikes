"""Sensitivity analyses v1.2 (added 2026-11-22 after review iteration 1). Reads outputs of
pipeline.py (events_long.csv) and pipeline_sensitivity_v1_1.py (sensitivity_events.csv); changes nothing.
S4. Onsets clamped at the window edge (-400 ms; no threshold crossing found): counts by series and
    goal, and S2 BR-minus-TB lag in the 120 strike series with impact recomputed excluding strikes in
    which TB or BR onset was clamped.
S5. 120 strike series, DP RMS and MDF change with impact excluding P5 (whose DP channel shows
    artifact values: implausible MDF and RMS spread), Wilcoxon.
Run: python pipeline_sensitivity_v1_2.py <out_dir>
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
df = pd.read_csv(f"{OUT}/events_long.csv"); R = {}
subs = sorted(df.subj.unique()); pid = {s: f"P{i+1}" for i, s in enumerate(subs)}
cl = df.assign(clamped=df.onset_ms <= -400)
R["S4_clamped_counts"] = {f"{n}_{c}": dict(clamped=int(g.clamped.sum()), total=int(len(g)))
                          for (n, c), g in cl.groupby(["n", "cond"])}
R["S4_clamped_by_muscle_all"] = {m: int(g.clamped.sum()) for m, g in cl.groupby("muscle")}
b = df[(df.n == 120) & (df.cond == "impact")].copy()
b["phase"] = np.where(b.frac <= 0.2, "early", np.where(b.frac >= 0.8, "late", "mid")); b = b[b.phase != "mid"]
w = b.pivot_table(index=["subj", "phase", "event"], columns="muscle", values="onset_ms")
keep = (w.TB > -400) & (w.BR > -400)
lag = (w.BR - w.TB)[keep].groupby(["subj", "phase"]).median().unstack()
R["S4_BR_minus_TB_excluding_clamped"] = dict(excluded_strikes=int((~keep).sum()), early=float(lag.early.median()),
    late=float(lag.late.median()), change=float((lag.late - lag.early).median()), p=float(wilcoxon(lag.late, lag.early).pvalue))
el = b[b.muscle == "DP"].groupby(["subj", "phase"])[["rms", "mdf_hz"]].median().unstack("phase")
el = el.drop(index=[s for s in subs if pid[s] == "P5"], errors="ignore")
rc = 100 * (el["rms"]["late"] / el["rms"]["early"] - 1); fc = 100 * (el["mdf_hz"]["late"] / el["mdf_hz"]["early"] - 1)
R["S5_DP_impact_excluding_P5"] = dict(n=int(len(el)), rms_change_pct=float(rc.median()), n_rms_up=int((rc > 0).sum()),
    p_rms=float(wilcoxon(el["rms"]["late"], el["rms"]["early"]).pvalue), mdf_change_pct=float(fc.median()),
    p_mdf=float(wilcoxon(el["mdf_hz"]["late"], el["mdf_hz"]["early"]).pvalue))
json.dump(R, open(f"{OUT}/results_sensitivity_v1_2.json", "w"), indent=1); print(json.dumps(R, indent=1))
