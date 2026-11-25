"""Within-record candidate search used to flag windows for visual review."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import correlate, find_peaks

def detect(z, events):
    # z contains the four 1 kHz envelopes normalized by each channel's 99th percentile.
    channels = gaussian_filter1d(z[:, :3], 12, axis=0)[::10]
    centers = [round(event / 10) for event in events]
    segments = np.array([channels[k-25:k+26] for k in centers if 25 <= k < len(channels)-26])
    patterns = segments - segments.mean(axis=1, keepdims=True)
    patterns = patterns.reshape(len(segments), -1)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    similarity = patterns @ patterns.T
    representative = np.argmax(np.median(similarity, axis=1))
    similar = similarity[:, representative] > .65
    template = np.median(segments[similar], axis=0)
    template -= template.mean(axis=0)
    template /= np.linalg.norm(template)
    numerator = sum(correlate(channels[:, j], template[:, j], mode='same') for j in range(3))
    local_mean = uniform_filter1d(channels, 51, axis=0)
    variance = uniform_filter1d(channels**2, 51, axis=0) - local_mean**2
    score = numerator / np.sqrt(np.maximum(51*np.maximum(0, variance).sum(axis=1), 1e-10))
    candidates = 10 * find_peaks(score, height=.42, distance=65, prominence=.25)[0]
    composite = z[:, :3].sum(axis=1)
    min_height = .28*np.median(composite[events])
    times = []
    for k in candidates:
        if k < 400 or k+200 >= len(z) or not events[0]-750 <= k <= events[-1]+750:
            continue
        peak = k-140 + np.argmax(composite[k-140:k+141])
        if 400 <= peak < len(z)-200 and composite[peak] >= min_height:
            times.append(peak)
    return np.asarray(times), score, np.median(np.diff(events)), similar.sum()
