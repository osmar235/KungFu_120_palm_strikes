"""Timing consistency and activation profile v1.3 (added 2026-11-23; aim 4 of the manuscript).
These measures were defined after inspection of the group ensemble envelopes (Fig. 2).
Same events, filters, envelopes and onset rule as pipeline_core.py; earlier outputs are unchanged.

A. Between-strike timing variability (1 and 10 strike series; per participant, task goal and muscle):
   sd_peak    SD across strikes of the muscle's envelope peak time relative to the strike reference (ms)
   sd_onset   SD across strikes of the muscle's onset time (ms; clamped onsets excluded)
   sd_lag     reference-free: SD across strikes of the muscle's peak time minus the TB peak time (ms),
              for DA, BR, DP; for TB, SD of TB peak minus DA peak
   Equal-strike check: the same statistics using, for each participant, only the first k strikes of the
   1 and 10 strike series in each task goal, k = smaller of the two strike counts.
   Replication: the same SDs in the first 20% of the 120 strike series (independent trials).
B. Activation profile (1 and 10 strike series): rise_time_ms (20 to 80% of the strike's own peak on the
   final rising edge) and onset_to_peak_ms, participant medians.
C. Peak-aligned ensembles: each strike's envelope aligned on the muscle's own peak (-300 to +200 ms),
   normalized from its 5th percentile to its peak; participant means saved for the figure.
Task goal contrasts: Wilcoxon signed rank on participant values (n = 9).
Run: python pipeline_timing_v1_3.py <data_root> <out_dir>
"""
import sys, os, json
if len(sys.argv) > 1: os.environ["EMG_ROOT"] = sys.argv[1]
from pipeline_core import *
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
pre, post = PARAMS["win_pre_ms"], PARAMS["win_post_ms"]
rows, ens = [], {}
for (s, n, cond), f in sorted(list_trials().items()):
    E = envelope(bandpass(read(f))); ev, _ = strike_events(E, n)
    for j, r in enumerate(ev):
        frac = j / (len(ev) - 1) if len(ev) > 1 else 0.0
        on = onsets(E, r)
        for c, m in enumerate(MUS):
            e = E[r - pre:r + post, c]; base = np.percentile(e, 5); p = int(np.argmax(e)); en = (e - base) / (e[p] - base)
            i80 = p
            while i80 > 0 and en[i80] > 0.8: i80 -= 1
            i20 = i80
            while i20 > 0 and en[i20] > 0.2: i20 -= 1
            rows.append(dict(subj=s, n=n, cond=cond, event=j, frac=frac, muscle=m, onset_ms=on[c][0], peak_ms=on[c][1],
                             clamped=on[c][0] <= -pre, rise_time_ms=max(i80 - i20, 1), onset_to_peak_ms=on[c][1] - on[c][0]))
            if n in (1, 10):
                pk = r - pre + p
                if pk - 300 >= 0 and pk + 200 <= len(E):
                    w = E[pk - 300:pk + 200, c]; w = (w - np.percentile(w, 5)) / (w.max() - np.percentile(w, 5))
                    ens.setdefault((cond, m, s), []).append(w)
d = apply_exclusions(pd.DataFrame(rows)); d.to_csv(f"{OUT}/timing_events.csv", index=False)
np.save(f"{OUT}/peak_aligned_ensembles.npy", {str(k): np.mean(v, 0) for k, v in ens.items()}, allow_pickle=True)

def variability(sub):
    out = []
    for (cond, s), g in sub.groupby(["cond", "subj"]):
        pkw = g.pivot_table(index=["n", "event"], columns="muscle", values="peak_ms")
        onw = g[~g.clamped].pivot_table(index=["n", "event"], columns="muscle", values="onset_ms")
        for m in MUS:
            if m not in pkw: continue
            lag = (pkw[m] - pkw["DA"]) if m == "TB" else (pkw[m] - pkw["TB"])
            out.append(dict(cond=cond, subj=s, muscle=m, n_strikes=len(pkw), sd_peak=pkw[m].std(), sd_onset=onw[m].std(), sd_lag=lag.std()))
    return pd.DataFrame(out)

def contrast(v, cols=("sd_peak", "sd_onset", "sd_lag")):
    res = {}
    for m in MUS:
        a = v[(v.cond == "impact") & (v.muscle == m)].set_index("subj"); b = v[(v.cond == "no_impact") & (v.muscle == m)].set_index("subj")
        a, b = a.align(b, join="inner")
        res[m] = {c: dict(impact=float(a[c].median()), no_impact=float(b[c].median()), n_lower_impact=int((a[c] < b[c]).sum()),
                          p=float(wilcoxon(a[c], b[c]).pvalue)) for c in cols}
    return res

R = {}
s110 = d[d.n.isin([1, 10])]
v_all = variability(s110); v_all.to_csv(f"{OUT}/timing_variability.csv", index=False)
R["A_variability_1_10"] = contrast(v_all)
R["A_strikes_per_participant"] = {c: dict(median=float(v_all[(v_all.cond == c) & (v_all.muscle == "TB")].n_strikes.median()),
                                          min=int(v_all[(v_all.cond == c) & (v_all.muscle == "TB")].n_strikes.min()),
                                          max=int(v_all[(v_all.cond == c) & (v_all.muscle == "TB")].n_strikes.max())) for c in ["impact", "no_impact"]}
# equal-strike check: first k strikes in chronological order (1 strike trial first, then the 10 strike series)
eq = []
for s, g in s110.groupby("subj"):
    k = min(g[g.cond == "impact"].groupby(["n", "event"]).ngroups, g[g.cond == "no_impact"].groupby(["n", "event"]).ngroups)
    for cond in ["impact", "no_impact"]:
        gc = g[g.cond == cond]; keys = sorted(gc.groupby(["n", "event"]).groups.keys())[:k]
        eq.append(gc.set_index(["n", "event"]).loc[keys].reset_index())
R["A_variability_equal_strikes"] = contrast(variability(pd.concat(eq)))
# replication in the first 20% of the 120 strike series
s120 = d[(d.n == 120) & (d.frac <= 0.2)]
R["A_variability_120_first20"] = contrast(variability(s120))
# B profile
prof = d[d.n.isin([1, 10]) & ~d.clamped].groupby(["cond", "subj", "muscle"])[["rise_time_ms", "onset_to_peak_ms"]].median().reset_index()
R["B_profile"] = contrast(prof.rename(columns={}), cols=("rise_time_ms", "onset_to_peak_ms"))
json.dump(R, open(f"{OUT}/results_timing_v1_3.json", "w"), indent=1)
for k in ["A_variability_1_10", "A_variability_equal_strikes", "A_variability_120_first20", "B_profile"]:
    print("==", k)
    for m, t in R[k].items():
        print(m, " | ".join(f"{c}: {v['impact']:.1f} vs {v['no_impact']:.1f} ({v['n_lower_impact']}/9 lower w/ impact, p={v['p']:.3f})" for c, v in t.items()))
print(R["A_strikes_per_participant"])
