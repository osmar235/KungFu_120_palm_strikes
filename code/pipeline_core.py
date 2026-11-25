"""
Palm strike EMG (Yau-Man Kung Fu), 9 participants, 1/10/120 alternating palm strikes,
with impact (maximal force against shield) and without impact (maximal speed).
EMG on the dominant (instrumented) arm: ch1 anterior deltoid (DA), ch2 triceps brachii (TB),
ch3 brachioradialis (BR), ch4 posterior deltoid (DP). 1000 Hz.

Run:  python pipeline.py <data_root> <out_dir>
All numbers reported in the manuscript are written to <out_dir>/results.json and CSV tables.
"""
import sys, os, glob, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, welch
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = os.environ.get("EMG_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
FS = 1000
MUS = ["DA", "TB", "BR", "DP"]
PARAMS = dict(bandpass=[20, 450], notch=[60, 120, 180, 240, 300, 360, 420], envelope_lp=15,
              onset_frac=0.20, win_pre_ms=400, win_post_ms=200, min_cycle_ms=500,
              min_event_rel_amp=0.40, spec_win_ms=[-250, 150], early_late_frac=0.20,
              max_ipsi_strikes_10=5, excluded_channels=[[5, 120, "DP"]])
# v2.1 (2026-11-24): reviewed events of the 10 strike series (short_event_decisions.csv), same criteria as the
# long-bout review: intervening bursts with a different muscle pattern excluded, phase substitutions documented.
# v2.0: (1) in the 10 strike series only the first 5 strikes of the instrumented arm are analyzed
# (10 prescribed alternating strikes); (2) the posterior deltoid of participant P5 is excluded from
# the 120 strike series (artifact).

# ---------------------------------------------------------------- I/O
def list_trials():
    """De-identified layout: ROOT/P<k>/<goal>_<n>.csv.gz (goal = impact or no_impact; n = 1, 10, 120)."""
    out = {}
    for f in sorted(glob.glob(ROOT + "/P*/*.csv.gz")):
        subj = f.split("/")[-2]; goal, n = os.path.basename(f)[:-7].rsplit("_", 1)
        out[(subj, int(n), goal)] = f
    return out

CURRENT = {"record": None}
_SHORT_DEC = None

def _short_decisions():
    """Reviewed events of the 10 strike series (short_event_decisions.csv, v2.1)."""
    global _SHORT_DEC
    if _SHORT_DEC is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "short_event_decisions.csv")
        _SHORT_DEC = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["record", "decision", "sample0"])
    return _SHORT_DEC

def _set_current(f):
    trials = list_trials(); subjects = sorted({k[0] for k in trials})
    for (subj, n, goal), path in trials.items():
        if path == f:
            CURRENT["record"] = f"P{subjects.index(subj) + 1}_{goal}_{n}"; return
    CURRENT["record"] = None

def read(f):
    _set_current(f)
    """Return the four EMG channels (DA, TB, BR, DP; volts at the amplifier output, 1000 Hz)."""
    return pd.read_csv(f).iloc[:, 1:5].to_numpy()

# ---------------------------------------------------------------- signal processing
def bandpass(x):
    lo, hi = PARAMS["bandpass"]
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], "band"); y = filtfilt(b, a, x, axis=0)
    for f0 in PARAMS["notch"]:
        bb, aa = iirnotch(f0, 30, FS); y = filtfilt(bb, aa, y, axis=0)
    return y

def envelope(y):
    b, a = butter(4, PARAMS["envelope_lp"] / (FS / 2)); return filtfilt(b, a, np.abs(y), axis=0)

def strike_events(E, n):
    """Ipsilateral strike events: peaks of the summed normalized DA+TB+BR envelope.
    Strikes alternate hands, so instrumented-arm strikes recur every ~1 s; the isolated
    DP (or BR) bursts between them are the retraction of this arm during the contralateral strike."""
    z = E / np.percentile(E, 99, axis=0); s = z[:, :3].sum(1)
    if n == 1:
        return [int(np.argmax(s))], z
    pk, pr = find_peaks(s, distance=PARAMS["min_cycle_ms"], prominence=0.25 * np.percentile(s, 99))
    pk = pk[s[pk] > PARAMS["min_event_rel_amp"] * np.median(s[pk])]
    pre, post = PARAMS["win_pre_ms"], PARAMS["win_post_ms"]
    pk = [int(p) for p in pk if p - pre >= 0 and p + post < len(s)]
    if n == 10:
        d = _short_decisions(); d = d[d.record == CURRENT["record"]]
        drop = set(d.loc[d.decision == "exclude", "sample0"]); add = set(d.loc[d.decision == "insert", "sample0"])
        pk = sorted((set(pk) - drop) | add)
        pk = pk[:PARAMS["max_ipsi_strikes_10"]]
    return pk, z

def apply_exclusions(df):
    """Drop excluded channels (PARAMS['excluded_channels']: [participant rank, series, muscle])."""
    subjects = sorted({k[0] for k in list_trials()})
    mask = pd.Series(False, index=df.index)
    for rank, n, m in PARAMS["excluded_channels"]:
        mask |= (df.subj == subjects[rank - 1]) & (df.n == n) & (df.muscle == m)
    return df[~mask]

def excluded_retraction(subj, n):
    subjects = sorted({k[0] for k in list_trials()})
    return any(subj == subjects[r - 1] and n == nn and m == "DP" for r, nn, m in PARAMS["excluded_channels"])

def onsets(E, ref):
    a, b = ref - PARAMS["win_pre_ms"], ref + PARAMS["win_post_ms"]; res = []
    for c in range(4):
        e = E[a:b, c]; base = np.percentile(e, 5); p = int(np.argmax(e))
        thr = base + PARAMS["onset_frac"] * (e[p] - base); i = p
        while i > 0 and e[i] > thr:
            i -= 1
        res.append((a + i - ref, a + p - ref))
    return res

def mdf(seg):
    f, P = welch(seg, FS, nperseg=200, noverlap=100, nfft=1024)
    cp = np.cumsum(P); return f[np.searchsorted(cp, cp[-1] / 2)]

