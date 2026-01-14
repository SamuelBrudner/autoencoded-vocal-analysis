import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from typing import Tuple, Dict
from pathlib import Path
from ava.preprocessing.utils import get_spec, _mel, _inv_mel
import matplotlib.pyplot as plt

def compute_mel_acf(audio_path: Path, p: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute mel-ACF and estimate 1/e decay timescale.
    
    Returns:
        lags: Correlation lags in seconds
        acf: Normalized ACF values
        tau_e: 1/e decay time in seconds
    """
    fs, audio = wavfile.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Use full audio duration
    duration = len(audio) / fs
    target_freqs = np.linspace(_mel(p['min_freq']), _mel(p['max_freq']), p['num_freq_bins'])
    target_freqs = _inv_mel(target_freqs)
    
    # Get spectrogram (log-mel)
    # We use a single long window or iterate if too large, but for timescale 
    # we want the temporal evolution of the mel bands.
    spec, _ = get_spec(0, duration, audio, p, fs=fs, target_freqs=target_freqs)
    
    # spec shape: (freq_bins, time_bins)
    # Compute ACF per band
    time_bins = spec.shape[1]
    dt = duration / time_bins
    
    all_acfs = []
    for band in range(spec.shape[0]):
        x = spec[band, :]
        x = x - np.mean(x)
        # Full correlation
        corr = correlate(x, x, mode='full')
        corr = corr[len(corr)//2:] # Take positive lags
        if corr[0] > 0:
            corr = corr / corr[0]
        all_acfs.append(corr)
    
    avg_acf = np.mean(all_acfs, axis=0)
    lags = np.arange(len(avg_acf)) * dt
    
    # Find 1/e decay
    target = 1.0 / np.exp(1)
    idx = np.where(avg_acf <= target)[0]
    if len(idx) > 0:
        tau_e = lags[idx[0]]
    else:
        tau_e = lags[-1]
        
    return lags, avg_acf, tau_e

def run_timescale_analysis(audio_dir: Path, p: Dict, output_dir: Path):
    """Analyze all wav files in directory and summarize."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_files = list(audio_dir.glob("*.wav"))
    
    results = []
    plt.figure(figsize=(10, 6))
    
    for wav_f in wav_files:
        lags, acf, tau = compute_mel_acf(wav_f, p)
        results.append({'file': wav_f.name, 'tau_e': tau})
        plt.plot(lags[:len(lags)//4], acf[:len(acf)//4], label=f"{wav_f.name} (τ={tau:.4f}s)")
    
    plt.axhline(1/np.exp(1), color='r', linestyle='--', label='1/e')
    plt.xlabel('Lag (s)')
    plt.ylabel('Normalized ACF')
    plt.title('Mel-ACF Characteristic Timescales')
    plt.legend()
    plt.savefig(output_dir / "acf_summary.png")
    plt.close()
    
    # Save results
    with open(output_dir / "timescales.txt", "w") as f:
        for res in results:
            f.write(f"{res['file']}: {res['tau_e']:.6f}s\n")
            
    taus = [r['tau_e'] for r in results]
    if taus:
        mean_tau = np.mean(taus)
        print(f"Mean characteristic timescale: {mean_tau:.4f}s")
        # Recommendation: window length ~ 3-4 * mean_tau, stride ~ mean_tau/2
        print(f"Recommended window length: {mean_tau*4:.3f}s")
        print(f"Recommended stride: {mean_tau/2:.3f}s")
