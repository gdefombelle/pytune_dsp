import numpy as np
import numba
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal.windows import blackman

#
# Fonctions d'assistance pour l'algorithme PFD (optimisées avec Numba)
#

@numba.jit(nopython=True)
def find_start(x):
    """
    Trouve l'indice de début approximatif de la note (après l'attaque transitoire).
    Traduction de la fonction MATLAB find_start.
    """
    limit = 0.01
    if len(x) == 0:
        return 0
    
    maxamp = np.max(np.abs(x))
    if maxamp == 0:
        return 0
        
    n = np.argmax(np.abs(x))
    
    # Si le pic est trop proche du début, on ne peut pas chercher avant.
    if n - 100 <= 0:
        return n

    slice_ = x[0:n-100]
    if slice_.size == 0:
        return n
        
    maxamp2 = np.max(np.abs(slice_))
    n2 = np.argmax(np.abs(slice_))

    # La logique MATLAB (n = n2-1) est étrange. 
    # Cette implémentation (n = n2) suit la logique de recherche 
    # de "l'échantillon max avant le pic précédent".
    while maxamp2 > limit * maxamp and n > 0:
        n = n2
        if n - 100 <= 0:
            break
        slice_ = x[0:n-100]
        if slice_.size == 0:
            break
        maxamp2 = np.max(np.abs(slice_))
        n2 = np.argmax(np.abs(slice_))
        
    return n


@numba.jit(nopython=True)
def find_all_peaks(data, maxmin):
    """
    Trouve tous les pics locaux (maxima ou minima).
    Traduction de find_all_peaks.
    """
    if data.size < 3:
        return np.array([0, data.size - 1], dtype=np.int64)
        
    it = np.sign(np.diff(data))
    
    # +1 car np.diff décale les indices
    indices_diff = np.where(np.diff(it) == -2 * np.sign(maxmin))[0] + 1
    indices = []

    # Vérifier si le premier point est un pic
    if it.size > 0 and it[0] == -maxmin:
        indices.append(0)
        
    indices.extend(indices_diff)
    
    # Le code MATLAB inclut toujours le dernier point.
    indices.append(data.size - 1)
    
    return np.array(indices, dtype=np.int64)


@numba.jit(nopython=True)
def findmaxs(data, n):
    """
    Trouve les 'n' plus grandes valeurs et leurs indices.
    Traduction de findmaxs.
    """
    if n <= 0 or data.size == 0:
        return np.empty(0, dtype=data.dtype), np.empty(0, dtype=np.int64)
    
    # Fait une copie pour ne pas modifier les données originales
    ampq = data.copy()
    
    # S'assure que n n'est pas plus grand que la taille des données
    n = min(n, data.size)
    
    peakv = np.zeros(n, dtype=data.dtype)
    peaki = np.zeros(n, dtype=np.int64)
    
    # Valeur de masquage
    min_val = np.min(ampq) - 1.0
    
    for i in range(n):
        mi = np.argmax(ampq)
        peakv[i] = ampq[mi]
        peaki[i] = mi
        ampq[mi] = min_val
        
    return peakv, peaki


@numba.jit(nopython=True)
def findpeaks(data, n):
    """
    Trouve les 'n' plus grandes valeurs et leurs indices.
    Traduction de findpeaks.
    """
    if n <= 0 or data.size == 0:
        return np.empty(0, dtype=data.dtype), np.empty(0, dtype=np.int64)
        
    ampq = data.copy()
    n = min(n, data.size)
    peakv = np.zeros(n, dtype=data.dtype)
    peaki = np.zeros(n, dtype=np.int64)
    
    # Seule différence avec findmaxs : la valeur de masquage
    min_val = np.min(ampq)
    
    for i in range(n):
        mi = np.argmax(ampq)
        peakv[i] = ampq[mi]
        peaki[i] = mi
        ampq[mi] = min_val
        
    return peakv, peaki


@numba.jit(nopython=True)
def peakint(three_points):
    """
    Interpolation parabolique (quadratique) d'un pic.
    Implémentation de `peakint` (non fournie dans le MATLAB).
    Prend 3 points (a, b, c) où 'b' est le pic.
    Retourne l'offset (décalage) par rapport au point central 'b'.
    """
    a, b, c = three_points[0], three_points[1], three_points[2]
    
    # Évite la division par zéro
    denominator = a - 2*b + c
    if np.abs(denominator) < 1e-9:
        return 0.0
        
    # Formule de l'offset du pic quadratique
    return 0.5 * (a - c) / denominator


@numba.jit(nopython=True)
def rec_find_oct(Xfftz, iz, fmax_idx, fmin_idx, lim_bins):
    """
    Fonction récursive pour trouver les pics significatifs.
    Traduction de rec_find_oct.
    """
    # fmax_idx et fmin_idx sont des indices relatifs à Xfftz
    # iz contient des indices de pics, relatifs à Xfftz
    
    # fi = find(iz<fmax & iz>fmin)
    iz_mask = (iz < fmax_idx) & (iz > fmin_idx)
    fi_indices_in_iz = np.where(iz_mask)[0]
    
    if fi_indices_in_iz.size == 0:
        return np.empty(0, dtype=np.int64)
        
    # [m,mi] = max(Xfftz(iz(fi)));
    iz_filtered = iz[fi_indices_in_iz]
    amps_slice = Xfftz[iz_filtered]
    
    if amps_slice.size == 0:
        return np.empty(0, dtype=np.int64)
        
    mi_in_slice = np.argmax(amps_slice)
    
    # mi = iz(fi(mi));
    mi_idx = iz_filtered[mi_in_slice] # C'est l'indice dans Xfftz
    
    res = []
    
    # if mi-fmin> lim & fmax-mi>lim
    if (mi_idx - fmin_idx > lim_bins) and (fmax_idx - mi_idx > lim_bins):
        res.append(mi_idx)
    
    # if mi-fmin>lim
    if mi_idx - fmin_idx > lim_bins:
        res1 = rec_find_oct(Xfftz, iz, mi_idx, fmin_idx, lim_bins)
        if res1.size > 0:
            res.extend(res1)
            
    # if fmax-mi>lim
    if fmax_idx - mi_idx > lim_bins:
        res2 = rec_find_oct(Xfftz, iz, fmax_idx, mi_idx, lim_bins)
        if res2.size > 0:
            res.extend(res2)
            
    return np.array(res, dtype=np.int64)


def determine_octave(x_data, fs):
    """
    Première étape : estimation grossière de F0 (octave).
    Traduction de determine_octave.
    """
    n_fft = 2**15
    # Utilise rfft (FFT pour signaux réels)
    Xfft = np.abs(np.fft.rfft(x_data, n=n_fft))
    
    # La fréquence de bin = fs / n_fft
    scale = fs / n_fft

    start_idx = int(round(20 / scale))
    end_idx = int(round(5000 / scale))
    
    # + 1e-9 pour éviter log(0)
    Xfftz = 20 * np.log10(Xfft[start_idx:end_idx] + 1e-9)

    iz_all_peaks = find_all_peaks(Xfftz, 1)
    
    lim_bins = 25.0 / scale
    
    # La fonction MATLAB originale avait des indices globaux/locaux confus.
    # Cette version corrigée utilise des indices locaux.
    fmax_idx = len(Xfftz) - 1
    fmin_idx = 0
    
    res_local = rec_find_oct(Xfftz, iz_all_peaks, fmax_idx, fmin_idx, lim_bins)
    
    # iz contient maintenant les indices des pics récursifs (relatifs à Xfftz)
    iz = np.sort(np.unique(res_local))
    if iz.size == 0:
        # Fallback si aucun pic n'est trouvé
        iz = iz_all_peaks
        if iz.size == 0:
             return 30.0 # Estimation par défaut

    num_of_peaks = 30
    
    # fi sont des indices dans 'iz'
    pv, fi = findpeaks(Xfftz[iz], num_of_peaks)
    fi = np.sort(fi)

    # qw = diff des indices de bin des pics
    qw = np.diff(iz[fi])
    
    # qwi = find(qw<200/scale);
    qwi = np.where(qw < (200.0 / scale))[0]
    if qwi.size > 0:
        qw = qw[qwi]
    
    if qw.size == 0:
        # Fallback si le filtrage ne laisse rien
         qw = np.diff(iz[fi])
         if qw.size == 0:
             return 30.0 # Estimation par défaut

    # [N,x] = hist(qw,20);
    N, x_edges = np.histogram(qw, bins=20)
    # np.histogram retourne les bords (edges), MATLAB retourne les centres
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    
    # ni = find(x>20/scale);
    ni = np.where(x_centers > (20.0 / scale))[0]
    
    if ni.size > 0:
        # [m,mi]=max(N(ni));
        mi = np.argmax(N[ni])
        oct_bins = x_centers[ni[mi]]
    else:
        # Fallback si le filtrage de plage est vide
        mi = np.argmax(N)
        oct_bins = x_centers[mi]
        
    # oct = x(ni(mi))*scale;
    oct = oct_bins * scale
    return oct


def getf1(x_data, fs):
    """
    Deuxième étape : estimation préliminaire de F0.
    Traduction de getf1.
    """
    # Étape 1 : Estimation d'Octave
    oct = determine_octave(x_data, fs)
    
    # Étape 2 : FFT haute résolution
    n_fft = 2**20
    Xfft = np.abs(np.fft.rfft(x_data, n=n_fft))
    scale = fs / n_fft
    
    # Range = major third (tierce majeure) 2^(4/12) ~= 1.26
    start_idx = int(np.ceil(oct * 0.8 / scale)) 
    end_idx = int(np.ceil(oct * 1.26 / scale)) # 1.25 dans MATLAB, 1.26 est plus proche
    
    if end_idx >= len(Xfft):
        end_idx = len(Xfft) - 1
    if start_idx >= end_idx:
        start_idx = end_idx - 2
    if start_idx < 0:
        start_idx = 0
        
    Xfft_slice = Xfft[start_idx:end_idx]
    if Xfft_slice.size == 0:
        return oct # Fallback

    # [m,mi]=max(Xfft)
    mi_rel = np.argmax(Xfft_slice)
    
    offset = 0.0
    # if mi>1 && mi < length(Xfft)
    if mi_rel > 0 and mi_rel < len(Xfft_slice) - 1:
        # [interpolated_maximum,offset]=peakint(Xfft(mi-1:mi+1))
        three_points = Xfft_slice[mi_rel-1 : mi_rel+2]
        offset = peakint(three_points)
        
    # f1 = (mi+offset+start-1)*scale;
    # (mi_rel + start_idx) = indice global (0-based)
    f1 = (mi_rel + start_idx + offset) * scale
    return f1


@numba.jit(nopython=True)
def _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk):
    """
    Fonction d'assistance pour calculer la courbe de tendance PFD.
    """
    trend = [] # Numba supporte les listes de scalaires
    
    # fz_est = kx.*f1.*sqrt(1+Best.*kx.*kx)./sqrt(1+Best);
    fz_est = kx * f1 * np.sqrt(1 + Best * kx**2) / np.sqrt(1 + Best)
    
    # fz_est = round(fz_est./scale)*scale;
    # L'arrondi aux bins de la FFT est important
    fz_est = np.round(fz_est / scale) * scale
    
    # for k=2:maxk (kx commence à 2, k_idx commence à 0)
    for k_idx in range(len(kx)):
        
        # fz_frame_min = fz_est(k-1)-f1d;
        fz_frame_min = fz_est[k_idx] - f1d
        fz_frame_max = fz_est[k_idx] + f1d

        # fi2 = find(fk(ti)>fz_frame_min & fk(ti)<=fz_frame_max);
        fk_ti = fk[ti]
        fi2_mask = (fk_ti > fz_frame_min) & (fk_ti <= fz_frame_max)
        fi2_indices_in_ti = np.where(fi2_mask)[0]

        if fi2_indices_in_ti.size > 0:
            # [m,mi] = max(amps(ti(fi2)));
            amps_slice = amps[ti[fi2_indices_in_ti]]
            mi_in_slice = np.argmax(amps_slice)
            
            # (ti(fi2(mi))-1)*scale (MATLAB 1-based)
            # -> fk[ti[fi2_indices_in_ti[mi_in_slice]]] (Python 0-based)
            actual_freq = fk[ti[fi2_indices_in_ti[mi_in_slice]]]
            
            # trend = [trend ... - fz_est(k-1)];
            trend.append(actual_freq - fz_est[k_idx])
        else:
            # Si aucun pic n'est trouvé dans la fenêtre, on ne fait rien
            pass 
            
    return np.array(trend)


@numba.jit(nopython=True)
def _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk):
    """
    Boucle d'affinement pour le coefficient B (inharmonicité).
    """
    Bm = 1.0
    i = 0
    # while i<40 & (i==0 | abs(Bm)>10^-4)
    while i < 40 and (i == 0 or np.abs(Bm) > 1e-4):
        i += 1
        trend = _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk)
        
        if len(trend) < 2: # Besoin d'au moins 2 points pour np.diff
            break 
            
        # rx = sum(sign(diff(trend)));
        rx = np.sum(np.sign(np.diff(trend)))
        
        # Logique de mise à jour adaptative
        if rx < 0:
            if Bm >= 0:
                Bm = -Bm / 2.0
        else:
            if Bm <= 0:
                Bm = -Bm / 2.0
        
        # Best = Best*10^Bm;
        Best = Best * (10**Bm)
        
    return Best


@numba.jit(nopython=True)
def _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk):
    """
    Boucle d'affinement pour la fréquence fondamentale F0.
    """
    fm = 0.005
    fsign = 0.0
    i = 0
    trend = np.empty(0, dtype=np.float64)
    
    # while i<100 & (i==0 | fm>10^-5)
    while i < 100 and (i == 0 or fm > 1e-5):
        i += 1
        trend = _calculate_trend(f1, Best, kx, scale, f1d, ti, fk, amps, maxk)
        
        if len(trend) == 0:
            break
            
        # qs = sign(mean(trend(1:ceil(length(trend)/2))));
        half_len = int(np.ceil(len(trend) / 2.0))
        if half_len == 0:
            break
        
        # np.mean sur un tableau vide renvoie NaN, ce qui est géré
        qs = np.sign(np.mean(trend[0:half_len]))
        
        # Logique de mise à jour adaptative
        if fsign == 0:
            fsign = qs
        elif fsign != qs:
            fm = fm / 2.0
            fsign = qs
        
        # f1 = f1*(1+fm*fsign);
        f1 = f1 * (1.0 + fm * fsign)
        
    return f1, trend # Retourne la dernière tendance pour le traçage


#
# Fonction principale PFD
#

def pfd(x_data, fs, plotflag=0, f1=None):
    """
    PFD - Algorithme d'estimation de l'inharmonicité (B) et de F0.
    
    Traduction Python/Numba du code pfd.m de J. Rauhala et V. Välimäki.
    
    Paramètres:
        x_data : signal d'entrée (vecteur NumPy)
        fs : fréquence d'échantillonnage (en Hz)
        plotflag : (optionnel) si 1, trace la courbe PFD finale
        f1 : (optionnel) F0 préliminaire externe. Si None, elle est estimée.
        
    Retourne:
        B : Coefficient d'inharmonicité estimé
        f1 : F0 estimée
    """
    
    # Étape 1: Estimation F0 préliminaire (si non fournie)
    if f1 is None:
        # Note : getf1 appelle determine_octave.
        f1 = getf1(x_data, fs)
    
    # Assure que le signal est un vecteur 1D
    x_data = x_data.ravel()

    # Étape 2: Fenêtrage du signal
    start = find_start(x_data)
    length = start + round(fs) # 1 seconde de signal
    if length > len(x_data):
        length = len(x_data)
        
    x_data_win = x_data[start:length] - np.mean(x_data[start:length])
    x_data_win = x_data_win * blackman(len(x_data_win))

    # Étape 3: FFT et sélection des pics
    n_fft = 2**16
    # Utilise rfft pour signaux réels
    Xfft = np.abs(np.fft.rfft(x_data_win, n=n_fft))
    fftlen = len(Xfft)
    scale = fs / n_fft # Fréquence par bin

    fk = np.arange(fftlen) * scale
    # +1e-9 pour éviter log(0)
    amps = 20. * np.log10(Xfft + 1e-9)
    
    # Sélection des pics proéminents (cf. article section 2.3 et code)
    winlen = round(f1 * 5)
    maxk = 50 # Nombre max de partiels

    q = int(np.ceil(20000 / winlen)) # Plafond à 20kHz
    q_maxk = int(np.ceil(maxk / 5.0))
    if q > q_maxk:
        q = q_maxk
        
    ti_list = []
    for i in range(1, q + 1):
        # fi = find(fk>(i-1)*winlen & fk<=i*winlen);
        fi = np.where((fk > (i - 1) * winlen) & (fk <= i * winlen))[0]
        if fi.size == 0:
            continue
            
        # piz = find_all_peaks(amps(fi)',1);
        piz = find_all_peaks(amps[fi], 1)
        if piz.size == 0:
            continue
            
        # [p,pi] = findmaxs(amps(fi(piz)),10);
        p, pi = findmaxs(amps[fi[piz]], 10)
        
        # ti = [ti fi(piz(pi))];
        ti_list.append(fi[piz[pi]])

    if not ti_list:
        print("PFD Warning: Aucun pic spectral trouvé. Retour F0 préliminaire.")
        return 0.0, f1

    ti = np.sort(np.concatenate(ti_list)) # ti = indices des pics spectraux
    
    # Étape 4: Boucles d'affinement (le cœur de l'algorithme PFD)
    # Celles-ci sont exécutées par les fonctions Numba-jitted.
    
    Best = 0.0001 # Estimation initiale de B
    kx = np.arange(2, maxk + 1, dtype=np.float64) # Indices des partiels (k=2, 3, ...)
    f1d = f1 * 0.4 # Largeur de la fenêtre de recherche

    # Le code MATLAB exécute B -> F0 -> B -> F0
    
    # 1. Affinement B
    Best = _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk)
    
    # 2. Affinement F0
    f1, _ = _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk)
    
    # 3. Second affinement B
    Best = _refine_b(Best, f1, f1d, kx, scale, ti, fk, amps, maxk)
    
    # 4. Second affinement F0
    f1, final_trend = _refine_f0(f1, Best, f1d, kx, scale, ti, fk, amps, maxk)
    
    B = Best

    # Étape 5: Traçage (optionnel)
    if plotflag == 1:
        plt.figure()
        plt.plot(final_trend, 'bo-')
        plt.title(f'Courbe PFD Finale (B={B:.2e}, F0={f1:.2f} Hz)')
        plt.xlabel('Indice de Partiel (relatif)')
        plt.ylabel('Déviation (Hz)')
        plt.grid(True)
        plt.show()

    return B, f1


def estimate_f0_pfd_numba(x: np.ndarray, fs: float, debug: bool = False) -> dict:
    """Interface PyTune-compatible (retour dict standard)."""
    B, f0 = pfd(x, fs, plotflag=int(debug))
    quality = float(np.clip(np.abs(B) * 1e6, 0, 1e6))  # indicateur simple
    return dict(
        f0=float(f0),
        B=float(B),
        quality=quality,
        deltas=np.array([]),
        partials_hz=np.array([]),
    )
#
# Exemple d'utilisation (bloc main)
#
if __name__ == '__main__':
    # Crée un signal de test synthétique (car 'mytone.wav' n'est pas fourni)
    # Ce signal est un son de piano inharmonique simulé.
    
    fs = 44100
    duration = 1.5
    t = np.arange(int(fs * duration)) / fs
    
    f0_true = 110.0  # A2
    B_true = 0.0005
    
    print(f"--- Test PFD avec signal synthétique ---")
    print(f"F0 réelle: {f0_true:.2f} Hz, B réel: {B_true:.2e}")

    x_synth = np.zeros_like(t)
    
    try:
        # Génération du son
        for k in range(1, 30):
            # Formule d'inharmonicité (Eq. 1 de l'article)
            # fk = f0 * k * sqrt(1 + B*k^2)
            # Note: Le code MATLAB utilise fz_est = kx.*f1.*sqrt(1+Best.*kx.*kx)./sqrt(1+Best);
            # Ce qui est une normalisation différente. Utilisons celle du code :
            fk = f0_true * k * np.sqrt(1 + B_true * k**2) / np.sqrt(1 + B_true)
            
            # Ajoute une décroissance exponentielle
            amplitude = np.exp(-t * 0.8 * k**0.5)
            x_synth += amplitude * np.sin(2 * np.pi * fk * t)
            
        # Ajoute un peu de bruit
        x_synth += np.random.randn(len(x_synth)) * 0.01
        
        # Normalisation
        x_synth = x_synth / np.max(np.abs(x_synth)) * 0.8
        
        print("\nEstimation en cours (plotflag=1)...")
        
        # Exécute l'algorithme PFD
        # Note: La première exécution de Numba inclut le temps de compilation.
        B_est, f1_est = pfd(x_synth, fs, plotflag=1)
        
        print("\n--- Résultats ---")
        print(f"F0 estimée: {f1_est:.2f} Hz (Erreur: {f1_est - f0_true:.2f} Hz)")
        print(f"B estimé: {B_est:.2e} (Erreur: {B_est - B_true:.2e})")

        # Exemple avec un fichier WAV (si vous en avez un)
        # try:
        #     fs_wav, x_wav = wavfile.read('votre_son_piano.wav')
        #     # Convertir en float64 normalisé
        #     if x_wav.dtype == np.int16:
        #         x_wav = x_wav.astype(np.float64) / 32768.0
        #     elif x_wav.dtype == np.float32:
        #         x_wav = x_wav.astype(np.float64)
            
        #     # Si stéréo, prendre un seul canal
        #     if x_wav.ndim > 1:
        #         x_wav = x_wav[:, 0]
                 
        #     print("\nEstimation sur fichier 'votre_son_piano.wav'...")
        #     B_wav, f1_wav = pfd(x_wav, fs_wav, plotflag=1)
        #     print(f"F0 estimée (WAV): {f1_wav:.2f} Hz, B estimé (WAV): {B_wav:.2e}")
            
        # except FileNotFoundError:
        #     print("\nFichier 'votre_son_piano.wav' non trouvé. Skip test fichier.")
            
    except Exception as e:
        print(f"\nUne erreur est survenue lors du test : {e}")
        import traceback
        traceback.print_exc()