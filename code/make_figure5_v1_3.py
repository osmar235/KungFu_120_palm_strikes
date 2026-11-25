"""Figure 5 (v1.3): peak-aligned ensembles by task goal. Reads out/peak_aligned_ensembles.npy and
out/results_timing_v1_3.json from pipeline_timing_v1_3.py.  Run: python make_figure5_v1_3.py <out_dir>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "DejaVu Sans"})
E = np.load(f"{OUT}/peak_aligned_ensembles.npy", allow_pickle=True).item()
R = json.load(open(f"{OUT}/results_timing_v1_3.json"))
assert R["B_profile"]["TB"]["rise_time_ms"]["impact"] == 74.5 and R["B_profile"]["TB"]["rise_time_ms"]["no_impact"] == 47.0
COL = {"DA": "#2a78d6", "TB": "#eb6834", "BR": "#1baf7a", "DP": "#eda100"}
NAME = {"DA": "Anterior deltoid", "TB": "Triceps brachii", "BR": "Brachioradialis", "DP": "Posterior deltoid"}
subs = sorted({eval(k)[2] for k in E}); assert len(subs) == 9
tt = np.arange(-300, 200)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
for k, m in enumerate(["TB", "BR", "DA", "DP"]):
    for cond, ls, col, lab in (("impact", "-", COL[m], "With impact"), ("no_impact", "--", "#52514e", "Without impact")):
        M = np.array([E[str((cond, m, s))] for s in subs]); mu = M.mean(0); se = M.std(0) / np.sqrt(len(subs))
        ax[k].plot(tt, mu, color=col, ls=ls, lw=2, label=lab); ax[k].fill_between(tt, mu - se, mu + se, color=col, alpha=.15, lw=0)
    rt = R["B_profile"][m]["rise_time_ms"]
    ax[k].text(-290, 0.93, f"rise time 20-80%\nimpact {rt['impact']:.1f} ms\nno impact {rt['no_impact']:.1f} ms", fontsize=9, color="#52514e", va="top")
    ax[k].axvline(0, color="#52514e", lw=.6); ax[k].set_title(NAME[m], fontsize=9, loc="left")
    ax[k].set_xlabel("time from the muscle's own peak (ms)")
ax[0].set_ylabel("normalized envelope\n(mean ± SE, n = 9)")
h, l = ax[0].get_legend_handles_labels(); hg = [plt.Line2D([], [], color="#52514e", lw=2, ls="-"), plt.Line2D([], [], color="#52514e", lw=2, ls="--")]
fig.legend(hg, ["With impact (colored, solid)", "Without impact (gray, dashed)"], loc="lower center", ncol=2, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f"{OUT}/fig5_peak_aligned.png", dpi=300)
print("ok")
