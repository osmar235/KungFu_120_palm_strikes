"""Rate of EMG rise v1.3 (added 2026-11-22 at the author's request, post hoc exploratory analysis
suggested by the group ensemble envelopes, Fig. 2). Same events, filters and reference as pipeline_core.py.
Per ipsilateral strike and muscle, within the window -400 to +200 ms (rising edge = from onset to peak):
  rise_time_ms   time from 20% to 80% of the strike's own peak above baseline (5th percentile) on the
                 rising edge (last 20% crossing before the 80% crossing before the peak)
  slope_norm     0.6 / rise_time (normalized units per second): rate of rise independent of amplitude
  dmax_norm      maximum first derivative of the peak-normalized envelope before the peak (1/s)
  rer50_raw      mean slope of the raw envelope over the first 50 ms after onset (V/s); compared only
                 within participant and muscle (no MVC normalization)
  onset_to_peak  peak time minus onset time (ms)
Task goal contrast on participant medians of the 1 and 10 strike series (Wilcoxon, uncorrected).
Within-bout change in the 120 strike series (first vs last 20%).
Run: python pipeline_rise_v1_3.py <data_root> <out_dir>
"""
import sys, os, json
if len(sys.argv) > 1: os.environ["EMG_ROOT"] = sys.argv[1]
from pipeline_core import *
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
pre, post = PARAMS["win_pre_ms"], PARAMS["win_post_ms"]
rows = []
for (s, n, cond), f in sorted(list_trials().items()):
    x = read(f); E = envelope(bandpass(x)); ev, _ = strike_events(E, n)
    for j, r in enumerate(ev):
        frac = j / (len(ev) - 1) if len(ev) > 1 else 0.0
        on = onsets(E, r)
        for c in range(4):
            e = E[r - pre:r + post, c]; base = np.percentile(e, 5); p = int(np.argmax(e)); amp = e[p] - base
            en = (e - base) / amp
            i80 = p
            while i80 > 0 and en[i80] > 0.8: i80 -= 1
            i20 = i80
            while i20 > 0 and en[i20] > 0.2: i20 -= 1
            rt = max(i80 - i20, 1)
            d = np.diff(en[:p + 1]) * FS if p > 1 else np.array([np.nan])
            o = on[c][0] + pre  # onset index in window
            o50 = min(o + 50, len(e) - 1)
            rows.append(dict(subj=s, n=n, cond=cond, event=j, frac=frac, muscle=MUS[c], rise_time_ms=rt,
                             slope_norm=0.6 / (rt / 1000), dmax_norm=float(np.nanmax(d)),
                             rer50_raw=float((e[o50] - e[o]) / ((o50 - o) / 1000)) if o50 > o else np.nan,
                             onset_to_peak_ms=on[c][1] - on[c][0], clamped=on[c][0] <= -pre))
d = apply_exclusions(pd.DataFrame(rows)); d.to_csv(f"{OUT}/rise_events.csv", index=False)
V = ["rise_time_ms", "slope_norm", "dmax_norm", "rer50_raw", "onset_to_peak_ms"]
R = {}
g = d[d.n.isin([1, 10]) & ~d.clamped].groupby(["cond", "subj", "muscle"])[V].median()
R["task_goal"] = {}
for m in MUS:
    a = g.xs(("impact", m), level=("cond", "muscle")); b = g.xs(("no_impact", m), level=("cond", "muscle"))
    R["task_goal"][m] = {v: dict(impact=float(a[v].median()), no_impact=float(b[v].median()),
                                 n_impact_higher=int((a[v] > b[v]).sum()), p=float(wilcoxon(a[v], b[v]).pvalue)) for v in V}
bb = d[(d.n == 120) & ~d.clamped].copy()
bb["phase"] = np.where(bb.frac <= 0.2, "early", np.where(bb.frac >= 0.8, "late", "mid")); bb = bb[bb.phase != "mid"]
h = bb.groupby(["cond", "subj", "muscle", "phase"])[V].median().unstack("phase")
R["bout"] = {}
for cond in ["impact", "no_impact"]:
    R["bout"][cond] = {}
    for m in MUS:
        x = h.loc[cond].xs(m, level="muscle")
        R["bout"][cond][m] = {v: dict(early=float(x[v]["early"].median()), late=float(x[v]["late"].median()),
                                      p=float(wilcoxon(x[v]["late"], x[v]["early"]).pvalue)) for v in ["slope_norm", "rer50_raw", "rise_time_ms"]}
json.dump(R, open(f"{OUT}/results_rise_v1_3.json", "w"), indent=1)
for m in MUS:
    t = R["task_goal"][m]
    print(m, " | ".join(f"{v}: imp {t[v]['impact']:.3g} vs no {t[v]['no_impact']:.3g} ({t[v]['n_impact_higher']}/9 imp>no, p={t[v]['p']:.3f})" for v in V))
print("BOUT")
for c in R["bout"]:
    for m in MUS:
        t = R["bout"][c][m]; print(c, m, " | ".join(f"{v}: {t[v]['early']:.3g}->{t[v]['late']:.3g} p={t[v]['p']:.3f}" for v in t))
