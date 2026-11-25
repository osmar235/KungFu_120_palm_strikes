"""Robustness of the timing-consistency result v1.4 (added 2026-11-23 after review of version 2).
Reads timing_events.csv written by pipeline_timing_v1_3.py; changes nothing else.
Between-strike dispersion of the reference-free inter-muscle peak lags (TB-DA, BR-TB, DP-TB) and of
peak times, using the median absolute deviation (MAD, scaled by 1.4826 to be comparable with an SD)
and the interquartile range (IQR) instead of the SD, for the 1 and 10 strike series, the equal-strike
subset and the first 20% of the 120 strike series. Wilcoxon signed rank, uncorrected.
Run: python pipeline_timing_robust_v1_4.py <out_dir>
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
d = pd.read_csv(f"{OUT}/timing_events.csv")
MAD = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
IQR = lambda x: np.subtract(*np.percentile(x, [75, 25]))
def disp(sub):
    rows = []
    for (cond, s), g in sub.groupby(["cond", "subj"]):
        pk = g.pivot_table(index=["n", "event"], columns="muscle", values="peak_ms")
        nanS = pd.Series(dtype=float)
        DP = pk["DP"] if "DP" in pk else pd.Series(np.nan, index=pk.index)
        series = {"lag_TB_DA": pk.TB - pk.DA, "lag_BR_TB": pk.BR - pk.TB, "lag_DP_TB": DP - pk.TB,
                  "peak_TB": pk.TB, "peak_DA": pk.DA, "peak_BR": pk.BR, "peak_DP": DP}
        for k, v in series.items():
            v = v.dropna().to_numpy()
            if len(v) < 2: continue
            rows.append(dict(cond=cond, subj=s, measure=k, sd=v.std(ddof=1), mad=MAD(v), iqr=IQR(v)))
    return pd.DataFrame(rows)
def contrast(t):
    out = {}
    for m, g in t.groupby("measure"):
        a = g[g.cond == "impact"].set_index("subj"); b = g[g.cond == "no_impact"].set_index("subj"); a, b = a.align(b, join="inner")
        out[m] = {st: dict(impact=float(a[st].median()), no_impact=float(b[st].median()), n_lower_impact=int((a[st] < b[st]).sum()),
                           p=float(wilcoxon(a[st], b[st]).pvalue)) for st in ["sd", "mad", "iqr"]}
    return out
s110 = d[d.n.isin([1, 10])]
eq = []
for s, g in s110.groupby("subj"):
    k = min(g[g.cond == "impact"].groupby(["n", "event"]).ngroups, g[g.cond == "no_impact"].groupby(["n", "event"]).ngroups)
    for cond in ["impact", "no_impact"]:
        gc = g[g.cond == cond]; keys = sorted(gc.groupby(["n", "event"]).groups.keys())[:k]
        eq.append(gc.set_index(["n", "event"]).loc[keys].reset_index())
R = {"series_1_10": contrast(disp(s110)), "equal_strikes": contrast(disp(pd.concat(eq))),
     "first20_of_120": contrast(disp(d[(d.n == 120) & (d.frac <= 0.2)]))}
json.dump(R, open(f"{OUT}/results_timing_robust_v1_4.json", "w"), indent=1)
for k, t in R.items():
    print("==", k)
    for m in ["lag_TB_DA", "lag_BR_TB", "lag_DP_TB", "peak_TB", "peak_DA"]:
        print(m, " | ".join(f"{st}: {v['impact']:.1f} vs {v['no_impact']:.1f} ({v['n_lower_impact']}/9, p={v['p']:.3f})" for st, v in t[m].items()))
