"""Figures for the palm strike EMG manuscript. Reads outputs of pipeline.py.
python make_figures.py <data_root> <out_dir>"""
import sys, os, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.argv = sys.argv[:3]
if len(sys.argv) > 1: os.environ["EMG_ROOT"] = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"



plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": False, "font.family": "DejaVu Sans"})
COL = {"DA": "#2a78d6", "TB": "#eb6834", "BR": "#1baf7a", "DP": "#eda100"}
LS = {"DA": "-", "TB": "-", "BR": "--", "DP": ":"}
MK = {"DA": "o", "TB": "s", "BR": "^", "DP": "D"}
NAME = {"DA": "Anterior deltoid", "TB": "Triceps brachii", "BR": "Brachioradialis", "DP": "Posterior deltoid"}
MUS = ["DA", "TB", "BR", "DP"]
COND = {"impact": "With impact (maximal force)", "no_impact": "Without impact (maximal speed)"}
df = pd.read_csv(f"{OUT}/events_long.csv")
_R = json.load(open(f"{OUT}/results.json"))
assert df.subj.nunique() == 9
assert _R["sequence"]["impact"]["median_onset_ms"] == {"DA": -88.0, "TB": -168.5, "BR": -100.5, "DP": -16.5}
assert _R["sequence"]["no_impact"]["median_onset_ms"] == {"DA": -46.5, "TB": -96.5, "BR": -68.5, "DP": -27.5}
assert _R["sequence"]["impact"]["n_subjects_TB_first"] == 8 and _R["sequence"]["no_impact"]["n_subjects_TB_first"] == 7
assert abs(_R["bout"]["impact"]["TB"]["rms_change_pct_median"] - 17.33) < 0.01
assert abs(_R["bout"]["impact"]["TB"]["mdf_change_pct_median"] + 11.82) < 0.01
ens = np.load(f"{OUT}/ensembles.npy", allow_pickle=True).item()
subs = sorted(df.subj.unique())
tt = np.arange(-400, 200)

# ---- Fig 1: alternating strikes, instrumented arm (raw envelopes), with and without impact
from pipeline_core import read, bandpass, envelope, list_trials, strike_events
T = list_trials()
fig, ax = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=True)
for a, cond in zip(ax, ["impact", "no_impact"]):
    x = read(T[(sorted({k[0] for k in T})[6], 120, cond)]); E = envelope(bandpass(x)); z = E / np.percentile(E, 99, axis=0)
    ev, _ = strike_events(E, 120)
    s0 = 20000; sl = slice(s0, s0 + 4000); t = np.arange(4000) / 1000
    for c, m in enumerate(MUS):
        a.plot(t, z[sl, c], color=COL[m], ls=LS[m], lw=1.4, label=NAME[m])
    for e in ev:
        if s0 <= e < s0 + 4000:
            a.axvline((e - s0) / 1000, color="#52514e", lw=0.6)
    a.set_title(COND[cond], loc="left", fontsize=9); a.set_ylabel("normalized envelope")
ax[1].set_xlabel("time (s)")
ax[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28))
ax[1].annotate("retraction burst\n(contralateral strike)", xy=(0.52, 1.0), xytext=(0.62, 1.35),
               fontsize=7, color="#52514e", arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.6))
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_alternating.png", dpi=300); plt.close()

# ---- Fig 2: group ensemble envelopes (1 and 10 strikes)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
R = json.load(open(f"{OUT}/results.json"))
for j, cond in enumerate(["impact", "no_impact"]):
    sm = []
    for s in subs:
        segs = [ens[str((s, n, cond))] for n in (1, 10) if str((s, n, cond)) in ens]
        sm.append(np.concatenate(segs).mean(0))
    sm = np.array(sm); m = sm.mean(0); se = sm.std(0) / np.sqrt(len(subs))
    for c, mu in enumerate(MUS):
        ax[j].plot(tt, m[:, c], color=COL[mu], ls=LS[mu], lw=2, label=NAME[mu])
        ax[j].fill_between(tt, m[:, c] - se[:, c], m[:, c] + se[:, c], color=COL[mu], alpha=.15, lw=0)
        on = R["sequence"][cond]["median_onset_ms"][mu]
        ax[j].plot([on], [-0.04], marker=MK[mu], color=COL[mu], ms=7, clip_on=False)
    ax[j].axvline(0, color="#52514e", lw=.6); ax[j].set_title(COND[cond], loc="left", fontsize=9)
    ax[j].set_xlabel("time relative to strike reference (ms)"); ax[j].set_ylim(-0.06, 1)
ax[0].set_ylabel("normalized envelope (mean ± SE, n = 9)")
ax[0].legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_group_envelopes.png", dpi=300); plt.close()

# ---- Fig 3: onset per subject
seq = pd.read_csv(f"{OUT}/onsets_subject_median.csv")
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
for j, cond in enumerate(["impact", "no_impact"]):
    d = seq[seq.cond == cond]
    for c, mu in enumerate(MUS):
        v = d[d.muscle == mu].set_index("subj").onset_ms.reindex(subs)
        ax[j].scatter(v, np.arange(9) + (c - 1.5) * 0.13, color=COL[mu], marker=MK[mu], s=30, label=NAME[mu], zorder=3,
                      edgecolor="#fcfcfb", linewidth=0.8)
        ax[j].axvline(v.median(), color=COL[mu], ls=LS[mu], lw=1.4, alpha=.8)
    ax[j].set_yticks(range(9)); ax[j].set_yticklabels([f"P{i+1}" for i in range(9)])
    ax[j].axvline(0, color="#52514e", lw=.6); ax[j].set_xlim(-300, 60)
    ax[j].set_title(COND[cond], loc="left", fontsize=9); ax[j].set_xlabel("activation onset (ms, participant median)")
ax[1].legend(fontsize=7, frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_onsets.png", dpi=300); plt.close()

# ---- Fig 4: 120-strike bout, RMS and MDF (median of participants, % of first 20%)
b = df[df.n == 120].copy(); b["bin"] = np.minimum((b.frac * 10).astype(int), 9)
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey="row")
for j, cond in enumerate(["impact", "no_impact"]):
    for r, v in enumerate(["rms", "mdf_hz"]):
        a = ax[r, j]
        for mu in MUS:
            d = b[(b.cond == cond) & (b.muscle == mu)].groupby(["subj", "bin"])[v].median().unstack()
            d = 100 * d.div(d[[0, 1]].mean(axis=1), axis=0); m = d.median(); x = (m.index + 0.5) * 10
            a.plot(x, m, color=COL[mu], ls=LS[mu], lw=2.2 if mu == "TB" else 1.3, marker=MK[mu], ms=4, label=NAME[mu])
            if mu == "TB":
                a.fill_between(x, d.quantile(.25), d.quantile(.75), color=COL[mu], alpha=.12, lw=0)
        a.axhline(100, color="#52514e", lw=.6)
        a.set_ylabel(("RMS amplitude" if v == "rms" else "Median frequency") + "\n(% of first 20% of bout)")
        if r == 0: a.set_title(COND[cond], loc="left", fontsize=9)
        if r == 1: a.set_xlabel("progress through the 120-strike bout (%)")
ax[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_bout.png", dpi=300); plt.close()
print("figures written")
