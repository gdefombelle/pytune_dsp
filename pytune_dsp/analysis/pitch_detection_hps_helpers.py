# === helpers notes / cents ====================================================
import numpy as np

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def freq_to_midi(f: float) -> float:
    if f <= 0: return float("nan")
    return 69.0 + 12.0*np.log2(f/440.0)

def midi_round_to_int(m: float) -> int:
    return int(round(m))

def midi_to_name(m_int: int) -> str:
    name = NOTE_NAMES[m_int % 12]
    octv = m_int // 12 - 1
    return f"{name}{octv}"

def cents_between(f: float, f_ref: float) -> float:
    if f <= 0 or f_ref <= 0: return float("nan")
    return 1200.0 * np.log2(f / f_ref)

def nearest_tempered_freq(f: float) -> tuple[int, float, str, float]:
    """Retourne (midi_int, f_temperé, note_name, cents_offset) pour f."""
    m = freq_to_midi(f)
    m_int = midi_round_to_int(m)
    f_eq = 440.0 * (2.0 ** ((m_int - 69) / 12.0))
    name = midi_to_name(m_int)
    cents_off = cents_between(f, f_eq)
    return m_int, f_eq, name, cents_off

# === afficheur clusters -> notes =============================================
def print_clusters_as_notes(clusters, *, title="Clusters → Notes tempérées"):
    """
    Affiche un tableau:
      idx | f_mean(Hz) | amp_sum | n | note_proche | MIDI | f_eq(Hz) | Δcents
    """
    print(f"\n{title}")
    print("-"*95)
    print(f"{'idx':>3} | {'f_mean(Hz)':>11} | {'amp_sum':>10} | {'n':>2} | {'note':>6} | {'MIDI':>4} | {'f_eq(Hz)':>10} | {'Δcents':>7}")
    print("-"*95)
    for i, c in enumerate(clusters):
        m, f_eq, name, dc = nearest_tempered_freq(c.f_mean)
        print(f"{i:>3} | {c.f_mean:>11.3f} | {c.amp_sum:>10.3f} | {c.n:>2} | {name:>6} | {m:>4} | {f_eq:>10.3f} | {dc:>7.2f}")
    print("-"*95)

# === test harmonique par rapport à f0 =========================================
def check_harmonic_consistency(clusters, f0: float, B: float = 0.0, *,
                               hmax: int = 16, cents_tol: float = 30.0):
    """
    Pour une f0 donnée (éventuellement avec inharmonicité B),
    liste les clusters qui matchent une harmonique h (|Δcents| ≤ cents_tol).
    """
    def f_h(h):
        return f0 * h * np.sqrt(1.0 + B*(h**2))

    matches = []   # (idx_cluster, h, Δcents, f_cluster, f_theory)
    misses  = []   # idx_cluster sans match

    for i, c in enumerate(clusters):
        # cherche l'h qui minimise |Δcents|
        cents_all = []
        for h in range(1, hmax+1):
            dc = cents_between(c.f_mean, f_h(h))
            cents_all.append((h, abs(dc), dc))
        h_best, best_abs, real_dc = min(cents_all, key=lambda t: t[1])
        if best_abs <= cents_tol:
            matches.append((i, h_best, real_dc, c.f_mean, f_h(h_best)))
        else:
            misses.append(i)

    # Affichage
    print(f"\nHarmonic check vs f0={f0:.3f} Hz  (B={B:.1e}, h≤{hmax}, tol=±{cents_tol}c)")
    print("-"*95)
    if matches:
        print(f"{'idx':>3} | {'h':>2} | {'f_cluster':>10} | {'f_theory':>10} | {'Δcents':>7} | note(cluster)")
        print("-"*95)
        for i,h,dc,fc,ft in matches:
            m, f_eq, name, off = nearest_tempered_freq(fc)
            print(f"{i:>3} | {h:>2} | {fc:>10.3f} | {ft:>10.3f} | {dc:>7.2f} | {name:>12}")
    else:
        print("(aucun match sous la tolérance)")

    if misses:
        print("-"*95)
        print("Clusters sans appariement harmonique (candidats bruit / autre note):", misses)
    print("-"*95)
    return matches, misses