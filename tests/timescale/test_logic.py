import numpy as np
from scipy.io import wavfile
from src.timescale_analysis import compute_mel_acf
from ava.preprocessing.utils import get_spec

NUM_FREQ_BINS = 128

def test_timescale_logic(tmp_path):
    # Create a dummy wav file with some structure (e.g. filtered noise)
    fs = 32000
    # Signal with ~50ms correlation
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(fs)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 20/(fs/2), btype='low')
    audio = lfilter(b, a, audio)

    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), fs, audio.astype(np.float32))
    
    p = {
        'fs': fs,
        'get_spec': get_spec,
        'num_freq_bins': NUM_FREQ_BINS,
        'num_time_bins': 256, # More bins for better resolution
        'nperseg': 512,
        'noverlap': 256,
        'max_dur': 2.0,
        'min_freq': 400,
        'max_freq': 10000,
        'spec_min_val': -25.0,
        'spec_max_val': 2.0,
        'mel': True,
        'time_stretch': False,
        'within_syll_normalize': False,
    }
    
    lags, acf, tau = compute_mel_acf(wav_path, p)
    print(f"Test Tau: {tau:.4f}s")
    assert tau > 0
    assert len(lags) == len(acf)

if __name__ == "__main__":
    test_timescale_logic()
