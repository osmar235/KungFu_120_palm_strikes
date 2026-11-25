"""Run: python pipeline.py <data_root> <out_dir>  (see pipeline_core.py for methods/parameters)"""
import sys, os, json
if len(sys.argv) > 1: os.environ["EMG_ROOT"] = sys.argv[1]
from pipeline_core import *
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
# ---------------------------------------------------------------- main extraction
trials = list_trials()
subjects = sorted({k[0] for k in trials})
rows, ens, retr, counts = [], {}, [], []
for (s, n, cond), f in sorted(trials.items()):
    x = read(f); y = bandpass(x); E = envelope(y)
    ev, z = strike_events(E, n)
    counts.append(dict(subj=s, n=n, cond=cond, n_events=len(ev), dur_s=len(x) / FS,
                       median_cycle_ms=float(np.median(np.diff(ev))) if len(ev) > 1 else np.nan))
    for j, r in enumerate(ev):
        frac = j / (len(ev) - 1) if len(ev) > 1 else 0.0
        on = onsets(E, r); a0, a1 = PARAMS["spec_win_ms"]
        seg = E[r - 400:r + 200]; nseg = (seg - np.percentile(seg, 5, 0)) / (seg.max(0) - np.percentile(seg, 5, 0))
        ens.setdefault((s, n, cond), []).append(nseg)
        for c in range(4):
            sg = y[r + a0:r + a1, c]
            post = nseg[400:600, c].mean()   # mean normalized envelope 0..200 ms after reference
            rows.append(dict(subj=s, n=n, cond=cond, event=j, frac=frac, t_s=r / FS, muscle=MUS[c],
                             onset_ms=on[c][0], peak_ms=on[c][1], rms=float(np.sqrt(np.mean(sg ** 2))),
                             mdf_hz=float(mdf(sg)), post_activity=float(post)))
        # retraction burst of DP between this and next ipsilateral strike
        if j + 1 < len(ev):
            a, b = r + int(0.2 * (ev[j + 1] - r)), r + int(0.8 * (ev[j + 1] - r))
            retr.append(dict(subj=s, n=n, cond=cond, event=j, frac=frac,
                             dp_retraction=float(z[a:b, 3].max()), dp_strike=float(z[r - 100:r + 150, 3].max())))
df = apply_exclusions(pd.DataFrame(rows)); df.to_csv(f"{OUT}/events_long.csv", index=False)
pd.DataFrame(counts).to_csv(f"{OUT}/event_counts.csv", index=False)
rt = pd.DataFrame(retr); rt = rt[[not excluded_retraction(a, b) for a, b in zip(rt.subj, rt.n)]]; rt.to_csv(f"{OUT}/retraction_dp.csv", index=False)
np.save(f"{OUT}/ensembles.npy", {str(k): np.array(v) for k, v in ens.items()}, allow_pickle=True)

R = {"params": PARAMS, "n_subjects": len(subjects), "subjects": subjects}
R["event_counts"] = pd.DataFrame(counts).groupby(["n", "cond"]).agg(
    events_median=("n_events", "median"), events_min=("n_events", "min"), events_max=("n_events", "max"),
    cycle_ms_median=("median_cycle_ms", "median")).reset_index().to_dict("records")

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * ps[k]); adj[k] = min(run, 1)
    return adj.tolist()

# ---------------------------------------------------------------- Aim 1/2: activation sequence (1 and 10 strikes, unfatigued)
seq = df[df.n.isin([1, 10])].groupby(["cond", "subj", "muscle"]).onset_ms.median().reset_index()
seq.to_csv(f"{OUT}/onsets_subject_median.csv", index=False)
R["sequence"] = {}
pairs = [("TB", "BR"), ("BR", "DA"), ("DA", "DP"), ("TB", "DA"), ("TB", "DP"), ("BR", "DP")]
for cond in ["impact", "no_impact"]:
    p = seq[seq.cond == cond].pivot(index="subj", columns="muscle", values="onset_ms")[MUS]
    fr = friedmanchisquare(*[p[m] for m in MUS])
    raw = [wilcoxon(p[a], p[b]).pvalue for a, b in pairs]
    ranks = p.rank(axis=1)
    R["sequence"][cond] = dict(
        median_onset_ms={m: float(p[m].median()) for m in MUS},
        iqr_onset_ms={m: [float(p[m].quantile(.25)), float(p[m].quantile(.75))] for m in MUS},
        mean_rank={m: float(ranks[m].mean()) for m in MUS},
        n_subjects_TB_first=int((ranks["TB"] == 1).sum()),
        n_subjects_DP_last=int((ranks["DP"] == 4).sum()),
        kendall_W=float(fr.statistic / (len(p) * (4 - 1))),
        friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
        pairwise={f"{a}<{b}": dict(median_diff_ms=float((p[a] - p[b]).median()), p_raw=float(pr), p_holm=float(ph))
                  for (a, b), pr, ph in zip(pairs, raw, holm(raw))})
# condition effect on onsets and on post-reference activity (braking index)
pa = df[df.n.isin([1, 10])].groupby(["cond", "subj", "muscle"])[["onset_ms", "post_activity"]].median().reset_index()
R["condition_effect"] = {}
for m in MUS:
    a = pa[(pa.cond == "impact") & (pa.muscle == m)].set_index("subj")
    b = pa[(pa.cond == "no_impact") & (pa.muscle == m)].set_index("subj")
    R["condition_effect"][m] = dict(
        onset_impact_ms=float(a.onset_ms.median()), onset_noimpact_ms=float(b.onset_ms.median()),
        onset_p=float(wilcoxon(a.onset_ms, b.onset_ms).pvalue),
        post_impact=float(a.post_activity.median()), post_noimpact=float(b.post_activity.median()),
        post_p=float(wilcoxon(a.post_activity, b.post_activity).pvalue))
# sequence spread (TB onset to DA onset) per condition
spread = seq.pivot_table(index=["cond", "subj"], columns="muscle", values="onset_ms")
spread["TB_to_DA"] = spread["DA"] - spread["TB"]
R["TB_to_DA_lag_ms"] = {c: float(spread.loc[c].TB_to_DA.median()) for c in ["impact", "no_impact"]}
R["TB_to_DA_lag_p"] = float(wilcoxon(spread.loc["impact"].TB_to_DA, spread.loc["no_impact"].TB_to_DA).pvalue)

# ---------------------------------------------------------------- Aim 3: 120-strike bouts
b = df[df.n == 120].copy(); q = PARAMS["early_late_frac"]
b["phase"] = np.where(b.frac <= q, "early", np.where(b.frac >= 1 - q, "late", "mid"))
el = b[b.phase != "mid"].groupby(["cond", "subj", "muscle", "phase"])[["rms", "mdf_hz", "onset_ms"]].median().unstack("phase")
el.to_csv(f"{OUT}/bout_early_late.csv")
R["bout"] = {}
for cond in ["impact", "no_impact"]:
    R["bout"][cond] = {}
    for m in MUS:
        d = el.loc[cond].xs(m, level="muscle")
        rc = 100 * (d["rms"]["late"] / d["rms"]["early"] - 1); fc = 100 * (d["mdf_hz"]["late"] / d["mdf_hz"]["early"] - 1)
        oc = d["onset_ms"]["late"] - d["onset_ms"]["early"]
        R["bout"][cond][m] = dict(
            rms_change_pct_median=float(rc.median()), rms_n_increase=int((rc > 0).sum()),
            rms_p=float(wilcoxon(d["rms"]["late"], d["rms"]["early"]).pvalue),
            mdf_early_hz=float(d["mdf_hz"]["early"].median()), mdf_late_hz=float(d["mdf_hz"]["late"].median()),
            mdf_change_pct_median=float(fc.median()), mdf_n_decrease=int((fc < 0).sum()),
            mdf_p=float(wilcoxon(d["mdf_hz"]["late"], d["mdf_hz"]["early"]).pvalue),
            onset_shift_ms_median=float(oc.median()),
            onset_p=float(wilcoxon(d["onset_ms"]["late"], d["onset_ms"]["early"]).pvalue))
# condition x time: compare % changes between conditions (paired by subject)
R["bout_condition_contrast"] = {}
for m in MUS:
    out = {}
    for v in ["rms", "mdf_hz"]:
        ci = el.loc["impact"].xs(m, level="muscle")[v]; si = el.loc["no_impact"].xs(m, level="muscle")[v]
        dci = 100 * (ci["late"] / ci["early"] - 1); dsi = 100 * (si["late"] / si["early"] - 1)
        out[v] = dict(impact_pct=float(dci.median()), noimpact_pct=float(dsi.median()), p=float(wilcoxon(dci, dsi).pvalue))
    R["bout_condition_contrast"][m] = out
# DP retraction burst across bout
r1 = rt[rt.n == 120].copy(); r1["phase"] = np.where(r1.frac <= q, "early", np.where(r1.frac >= 1 - q, "late", "mid"))
rr = r1.groupby(["cond", "subj"])[["dp_retraction", "dp_strike"]].median()
R["dp_retraction"] = {c: dict(retraction_median=float(rr.loc[c].dp_retraction.median()),
                              strike_median=float(rr.loc[c].dp_strike.median()),
                              ratio_retraction_over_strike=float((rr.loc[c].dp_retraction / rr.loc[c].dp_strike).median()))
                      for c in ["impact", "no_impact"]}
a, bb = rr.loc["impact"], rr.loc["no_impact"]
R["dp_retraction"]["ratio_p"] = float(wilcoxon(a.dp_retraction / a.dp_strike, bb.dp_retraction / bb.dp_strike).pvalue)

json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps(R, indent=1))
