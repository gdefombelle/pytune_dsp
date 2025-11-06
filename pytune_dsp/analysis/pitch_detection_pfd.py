# pitch_detection_pfd.py (version corrigée et enrichie)
import os
import numpy as np
import numba
from scipy.signal.windows import blackman
import matplotlib.pyplot as plt

# ==== Debug switch (0/1 via env) ==============================================
PFD_DEBUG_OCTAVE = bool(int(os.getenv("PFD_DEBUG_OCTAVE", "1")))
def _pfd_dbg(msg: str):
    if PFD_DEBUG_OCTAVE:
        print(msg)

# ==== Utilitaire : conversion Hz → note/octave ================================
def _note_from_freq(freq: float) -> str:
    if freq <= 0:
        return "?"
    midi = 69 + 12 * np.log2(freq / 440.0)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = int(round(midi)) % 12
    octave = int((round(midi) // 12) - 1)
    return f"{note_names[idx]}{octave}"

# ==== Fonctions auxiliaires Numba ============================================
@numba.jit(nopython=True)
def find_start(x):
    limit = 0.01
    if len(x) == 0:
        return 0
    maxamp = np.max(np.abs(x))
    if maxamp == 0:
        return 0
    n = np.argmax(np.abs(x))
    if n - 100 <= 0:
        return n
    slice_ = x[0:n-100]
    if slice_.size == 0:
        return n
    maxamp2 = np.max(np.abs(slice_))
    n2 = np.argmax(np.abs(slice_))
    while maxamp2 > limit * maxamp and n > 0:
        n = n2
        if n - 100 <= 0:
            break
        slice_ = x[0:n-100]
        if slice_.size <= 1:
            break
        maxamp2 = np.max(np.abs(slice_))
        n2 = np.argmax(np.abs(slice_))
    return n

@numba.jit(nopython=True)
def find_all_peaks(data, maxmin):
    if data.size < 3:
        return np.array([0, data.size - 1], dtype=np.int64)
    it = np.sign(np.diff(data))
    indices_diff = np.where(np.diff(it) == -2 * np.sign(maxmin))[0] + 1
    indices = []
    if it.size > 0 and it[0] == -maxmin:
        indices.append(0)
    indices.extend(indices_diff)
    indices.append(data.size - 1)
    return np.array(indices, dtype=np.int64)

@numba.jit(nopython=True)
def findmaxs(data, n):
    if n <= 0 or data.size == 0:
        return np.empty(0, dtype=data.dtype), np.empty(0, dtype=np.int64)
    ampq = data.copy()
    n = min(n, data.size)
    peakv = np.zeros(n, dtype=data.dtype)
    peaki = np.zeros(n, dtype=np.int64)
    min_val = np.min(ampq) - 1.0
    for i in range(n):
        mi = np.argmax(ampq)
        peakv[i] = ampq[mi]
        peaki[i] = mi
        ampq[mi] = min_val
    return peakv, peaki

@numba.jit(nopython=True)
def findpeaks(data, n):
    if n <= 0 or data.size == 0:
        return np.empty(0, dtype=data.dtype), np.empty(0, dtype=np.int64)
    ampq = data.copy()
    n = min(n, data.size)
    peakv = np.zeros(n, dtype=data.dtype)
    peaki = np.zeros(n, dtype=np.int64)
    min_val = np.min(ampq)
    for i in range(n):
        mi = np.argmax(ampq)
        peakv[i] = ampq[mi]
        peaki[i] = mi
        ampq[mi] = min_val
    return peakv, peaki

@numba.jit(nopython=True)
def peakint(three_points):
    a, b, c = three_points[0], three_points[1], three_points[2]
    denominator = a - 2*b + c
    if np.abs(denominator) < 1e-9:
        return 0.0
    return 0.5 * (a - c) / denominator

@numba.jit(nopython=True)
def rec_find_oct(Xfftz, iz, fmax_idx, fmin_idx, lim_bins):
    iz_mask = (iz < fmax_idx) & (iz > fmin_idx)
    fi_indices_in_iz = np.where(iz_mask)[0]
    if fi_indices_in_iz.size == 0:
        return np.empty(0, dtype=np.int64)
    iz_filtered = iz[fi_indices_in_iz]
    amps_slice = Xfftz[iz_filtered]
    if amps_slice.size == 0:
        return np.empty(0, dtype=np.int64)
    mi_in_slice = np.argmax(amps_slice)
    mi_idx = iz_filtered[mi_in_slice]
    res = []
    if (mi_idx - fmin_idx > lim_bins) and (fmax_idx - mi_idx > lim_bins):
        res.append(mi_idx)
    if mi_idx - fmin_idx > lim_bins:
        res1 = rec_find_oct(Xfftz, iz, mi_idx, fmin_idx, lim_bins)
        if res1.size > 0:
            res.extend(res1)
    if fmax_idx - mi_idx > lim_bins:
        res2 = rec_find_oct(Xfftz, iz, fmax_idx, mi_idx, lim_bins)
        if res2.size > 0:
            res.extend(res2)
    return np.array(res, dtype=np.int64)

# ==== Étape 1 : détermination grossière ======================================
def determine_octave(x_data, fs):
    n_fft = 2**15
    Xfft = np.abs(np.fft.rfft(x_data, n=n_fft))
    scale = fs / n_fft
    start_idx = int(round(20 / scale))
    end_idx = int(round(5000 / scale))
    Xfftz = 20 * np.log10(Xfft[start_idx:end_idx] + 1e-9)
    iz_all_peaks = find_all_peaks(Xfftz, 1)
    lim_bins = 25.0 / scale
    fmax_idx = len(Xfftz) - 1
    fmin_idx = 0
    res_local = rec_find_oct(Xfftz, iz_all_peaks, fmax_idx, fmin_idx, lim_bins)
    iz = np.sort(np.unique(res_local))
    if iz.size == 0:
        iz = iz_all_peaks
        if iz.size == 0:
            _pfd_dbg("[PFD octave] ❌ Aucun pic → fallback 30 Hz")
            return 30.0
    pv, fi = findpeaks(Xfftz[iz], 30)
    fi = np.sort(fi)
    qw = np.diff(iz[fi])
    qwi = np.where(qw < (200.0 / scale))[0]
    if qwi.size > 0:
        qw = qw[qwi]
    if qw.size == 0:
        qw = np.diff(iz[fi])
        if qw.size == 0:
            return 30.0
    N, x_edges = np.histogram(qw, bins=20)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    ni = np.where(x_centers > (20.0 / scale))[0]
    if ni.size > 0:
        mi = np.argmax(N[ni])
        oct_bins = x_centers[ni[mi]]
    else:
        mi = np.argmax(N)
        oct_bins = x_centers[mi]
    oct = oct_bins * scale
    note_str = _note_from_freq(oct)
    _pfd_dbg(f"[PFD octave] ✅ {len(iz)} pics | hist max={oct:.2f} Hz ≈ {note_str}")
    return oct

# ==== Étape 2 : estimation fine F1 ===========================================
def getf1(x_data, fs):
    oct = determine_octave(x_data, fs)
    n_fft = 2**20
    Xfft = np.abs(np.fft.rfft(x_data, n=n_fft))
    scale = fs / n_fft
    start_idx = int(np.ceil(oct * 0.8 / scale))
    end_idx = int(np.ceil(oct * 1.26 / scale))
    end_idx = min(end_idx, len(Xfft) - 1)
    start_idx = max(0, min(start_idx, end_idx - 2))
    Xfft_slice = Xfft[start_idx:end_idx]
    if Xfft_slice.size == 0:
        return oct
    mi_rel = np.argmax(Xfft_slice)
    offset = 0.0
    if 0 < mi_rel < len(Xfft_slice) - 1:
        offset = peakint(Xfft_slice[mi_rel-1:mi_rel+2])
    f1 = (mi_rel + start_idx + offset) * scale
    _pfd_dbg(f"[PFD f1] ✅ f1 préliminaire = {f1:.2f} Hz ≈ {_note_from_freq(f1)} (octave init {oct:.2f} Hz)")
    return f1

# ==== Étape 3 : algorithme PFD complet =======================================
@numba.jit(nopython=True)
def _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk):
    trend = []
    fz_est = kx * f1 * np.sqrt(1 + Best * kx**2) / np.sqrt(1 + Best)
    fz_est = np.round(fz_est / scale) * scale
    for k_idx in range(len(kx)):
        fz_frame_min = fz_est[k_idx] - f1d
        fz_frame_max = fz_est[k_idx] + f1d
        fk_ti = fk[ti]
        fi2_mask = (fk_ti > fz_frame_min) & (fk_ti <= fz_frame_max)
        fi2_indices_in_ti = np.where(fi2_mask)[0]
        if fi2_indices_in_ti.size > 0:
            amps_slice = amps[ti[fi2_indices_in_ti]]
            mi_in_slice = np.argmax(amps_slice)
            actual_freq = fk[ti[fi2_indices_in_ti[mi_in_slice]]]
            trend.append(actual_freq - fz_est[k_idx])
    return np.array(trend)

@numba.jit(nopython=True)
def _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk):
    Bm = 1.0
    i = 0
    while i < 40 and (i == 0 or np.abs(Bm) > 1e-4):
        i += 1
        trend = _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk)
        if len(trend) < 2:
            break
        rx = np.sum(np.sign(np.diff(trend)))
        if rx < 0:
            if Bm >= 0:
                Bm = -Bm / 2.0
        else:
            if Bm <= 0:
                Bm = -Bm / 2.0
        Best = Best * (10**Bm)
    return Best

@numba.jit(nopython=True)
def _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk):
    fm = 0.005
    fsign = 0.0
    i = 0
    trend = np.empty(0, dtype=np.float64)
    while i < 100 and (i == 0 or fm > 1e-5):
        i += 1
        trend = _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk)
        if len(trend) == 0:
            break
        half_len = int(np.ceil(len(trend) / 2.0))
        if half_len == 0:
            break
        qs = np.sign(np.mean(trend[0:half_len]))
        if fsign == 0:
            fsign = qs
        elif fsign != qs:
            fm = fm / 2.0
            fsign = qs
        f1 = f1 * (1.0 + fm * fsign)
    return f1, trend

def pfd(x_data, fs, plotflag=0, f1=None):
    if f1 is None:
        f1 = getf1(x_data, fs)
    x_data = x_data.ravel()
    start = find_start(x_data)
    length = min(start + round(fs), len(x_data))
    x_data_win = x_data[start:length] - np.mean(x_data[start:length])
    x_data_win *= blackman(len(x_data_win))
    n_fft = 2**16
    Xfft = np.abs(np.fft.rfft(x_data_win, n=n_fft))
    scale = fs / n_fft
    fk = np.arange(len(Xfft)) * scale
    amps = 20. * np.log10(Xfft + 1e-9)
    winlen = round(f1 * 5)
    maxk = 50
    q = int(np.ceil(20000 / winlen))
    q = min(q, int(np.ceil(maxk / 5.0)))
    ti_list = []
    for i in range(1, q + 1):
        fi = np.where((fk > (i - 1) * winlen) & (fk <= i * winlen))[0]
        if fi.size == 0:
            continue
        piz = find_all_peaks(amps[fi], 1)
        if piz.size == 0:
            continue
        p, pi = findmaxs(amps[fi[piz]], 10)
        ti_list.append(fi[piz[pi]])
    if not ti_list:
        _pfd_dbg("PFD Warning: Aucun pic spectral trouvé.")
        return 0.0, f1
    ti = np.sort(np.concatenate(ti_list))
    Best = 0.0001
    kx = np.arange(2, maxk + 1, dtype=np.float64)
    f1d = f1 * 0.4
    Best = _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk)
    f1, _ = _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk)
    Best = _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk)
    f1, trend = _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk)
    if plotflag == 1:
        plt.figure()
        plt.plot(trend, 'bo-')
        plt.title(f"PFD finale (B={Best:.2e}, F0={f1:.2f} Hz)")
        plt.grid(True)
        plt.show()
    _pfd_dbg(f"[PFD summary] f0_final={f1:.2f} Hz ≈ {_note_from_freq(f1)} | B={Best:.2e}")
    return Best, f1

# ==== Wrapper PyTune-compatible ==============================================
def estimate_f0_pfd_numba(x: np.ndarray, fs: float, debug: bool = False, f0_seed: float | None = None, expected_freq: float | None = None) -> dict:
    B, f0_pfd = pfd(x, fs, plotflag=int(debug))
    anchor = f0_seed or expected_freq
    if anchor and anchor > 0:
        ratios = np.array([0.5, 1.0, 2.0, 4.0])
        candidates = anchor * ratios
        k = np.argmin(np.abs(candidates - f0_pfd))
        f0_corr = candidates[k]
        if abs(f0_corr - f0_pfd) / max(f0_pfd, 1e-6) > 0.15:
            _pfd_dbg(f"[PFD octave-corr] {f0_pfd:.2f} Hz → {f0_corr:.2f} Hz (ratio {f0_corr/f0_pfd:.2f}×)")
        f0_pfd = f0_corr
        low, high = 0.6 * anchor, 1.6 * anchor
        if not (low <= f0_pfd <= high):
            f0_pfd = np.clip(f0_pfd, low, high)
    quality = float(np.clip(np.abs(B) * 1e6, 0, 1e6))
    unusable = (np.isnan(f0_pfd) or f0_pfd <= 0) or (np.isinf(B) or np.isnan(B)) or (quality >= 1e6)
    if unusable:
        return dict(f0=0.0, B=0.0, quality=quality, deltas=np.array([]), partials_hz=np.array([]))
    return dict(f0=float(f0_pfd), B=float(B), quality=quality, deltas=np.array([]), partials_hz=np.array([]))