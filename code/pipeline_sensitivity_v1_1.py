"""Sensitivity analyses v1.1 (added 2026-11-22 after independent review, iteration 0).
Does not change any v1.0 output. Same events, filters and reference as pipeline_core.py.

S1. Braking index without peak normalization: for each ipsilateral strike and muscle,
    ratio = mean raw envelope from 0 to 200 ms after the reference / mean raw envelope
    from 200 ms before to the reference (both from the same strike; no peak scaling).
    Task goal effect tested on participant medians (1 and 10 strike series), Wilcoxon.
S2. Reference-free timing in the 120 strike series: onset lag of each muscle relative to
    the triceps onset (muscle onset minus TB onset) and the TB peak-to-reference interval,
    first vs last 20% of ipsilateral strikes, Wilcoxon (participant medians).
S3. Strike-by-strike onset order: share of ipsilateral strikes (1 and 10 strike series)
    in which TB was the first of the four muscles, per participant.
Run: python pipeline_sensitivity_v1_1.py <data_root> <out_dir>
"""
import sys, os, json
if len(sys.argv) > 1: os.environ["EMG_ROOT"] = sys.argv[1]
from pipeline_core import *
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
trials = list_trials(); rows = []
for (s, n, cond), f in sorted(trials.items()):
    x = read(f); E = envelope(bandpass(x)); ev, _ = strike_events(E, n)
    for j, r in enumerate(ev):
        frac = j / (len(ev) - 1) if len(ev) > 1 else 0.0
        on = onsets(E, r)
        for c in range(4):
            post = E[r:r + 200, c].mean(); pre = E[r - 200:r, c].mean()
            rows.append(dict(subj=s, n=n, cond=cond, event=j, frac=frac, muscle=MUS[c],
                             onset_ms=on[c][0], peak_ms=on[c][1], post_pre_ratio=post / pre))
d = apply_exclusions(pd.DataFrame(rows)); d.to_csv(f"{OUT}/sensitivity_events.csv", index=False)
R = {}
# S1
a = d[d.n.isin([1, 10])].groupby(["cond", "subj", "muscle"]).post_pre_ratio.median().unstack("cond")
R["S1_post_pre_ratio"] = {m: dict(impact=float(a.xs(m, level="muscle").impact.median()),
                                  no_impact=float(a.xs(m, level="muscle").no_impact.median()),
                                  n_higher_no_impact=int((a.xs(m, level="muscle").no_impact > a.xs(m, level="muscle").impact).sum()),
                                  p=float(wilcoxon(a.xs(m, level="muscle").impact, a.xs(m, level="muscle").no_impact).pvalue))
                          for m in MUS}
# S2
b = d[d.n == 120].copy(); q = PARAMS["early_late_frac"]
b["phase"] = np.where(b.frac <= q, "early", np.where(b.frac >= 1 - q, "late", "mid")); b = b[b.phase != "mid"]
w = b.pivot_table(index=["cond", "subj", "phase", "event"], columns="muscle", values="onset_ms")
pk = b[b.muscle == "TB"].set_index(["cond", "subj", "phase", "event"]).peak_ms
R["S2_relative_to_TB_onset"] = {}
for cond in ["impact", "no_impact"]:
    out = {}
    for m in ["DA", "BR", "DP"]:
        lag = (w[m] - w["TB"]).loc[cond].groupby(["subj", "phase"]).median().unstack().dropna()
        out[f"{m}_minus_TB"] = dict(early=float(lag.early.median()), late=float(lag.late.median()),
                                    change=float((lag.late - lag.early).median()), p=float(wilcoxon(lag.late, lag.early).pvalue))
    tp = pk.loc[cond].groupby(["subj", "phase"]).median().unstack()
    out["TB_peak_rel_reference"] = dict(early=float(tp.early.median()), late=float(tp.late.median()),
                                         change=float((tp.late - tp.early).median()), p=float(wilcoxon(tp.late, tp.early).pvalue))
    R["S2_relative_to_TB_onset"][cond] = out
# S3
w1 = d[d.n.isin([1, 10])].pivot_table(index=["cond", "subj", "n", "event"], columns="muscle", values="onset_ms")
first = (w1[["DA", "TB", "BR", "DP"]].idxmin(axis=1) == "TB").groupby(["cond", "subj"]).mean().unstack("cond")
R["S3_share_strikes_TB_first"] = {c: dict(median=float(first[c].median()), min=float(first[c].min()), max=float(first[c].max()),
                                          n_strikes=int(w1.loc[c].shape[0])) for c in ["impact", "no_impact"]}
# full order counts on participant medians (v1.0 table), for the text
s = pd.read_csv(f"{OUT}/onsets_subject_median.csv")
R["full_order_TB_BR_DA_DP_participants"] = {}
for c in ["impact", "no_impact"]:
    p = s[s.cond == c].pivot(index="subj", columns="muscle", values="onset_ms")
    R["full_order_TB_BR_DA_DP_participants"][c] = int(((p.TB < p.BR) & (p.BR < p.DA) & (p.DA < p.DP)).sum())
json.dump(R, open(f"{OUT}/results_sensitivity.json", "w"), indent=1)
print(json.dumps(R, indent=1))
