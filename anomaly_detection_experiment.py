import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sax import SAX  


# Generate Training and Test Time Series
def generate_noisy_sine(length=1000, freq=5, noise_std=0.1, phase=0):
    """
    Generate a sine wave of given length and frequency with added Gaussian noise.
    """
    t = np.linspace(0, 1, length)
    return np.sin(2 * np.pi * freq * t + phase) + np.random.normal(0, noise_std, length)


def inject_anomalies(series, anomaly_positions, anomaly_width=20, freq=5, freq_change=2, phase_shift=np.pi/2):
    """
    Inject anomalies into the series at specified positions by modifying the local pattern
    (e.g., changing the frequency and phase).
    """
    series_anom = series.copy()
    length = len(series)
    t = np.linspace(0, 1, length)
    for pos in anomaly_positions:
        start = max(pos - anomaly_width // 2, 0)
        end = min(pos + anomaly_width // 2, length)
        series_anom[start:end] = np.sin(2 * np.pi * (freq + freq_change) * t[start:end] + phase_shift)
    return series_anom

# Markov Model on Symbolic Representations

def build_markov_model(symbols):
    """
    Build a first-order Markov chain model from a symbolic string.
    Returns a dictionary mapping each symbol to a dictionary of transition probabilities.
    """
    counts = {}
    total = {}
    for i in range(len(symbols)-1):
        s = symbols[i]
        s_next = symbols[i+1]
        if s not in counts:
            counts[s] = {}
            total[s] = 0
        counts[s][s_next] = counts[s].get(s_next, 0) + 1
        total[s] += 1
    model = {}
    for s in counts:
        model[s] = {s_next: count/total[s] for s_next, count in counts[s].items()}
    return model

def compute_anomaly_score(symbols, model, window_size=10):
    """
    For each sliding window in the symbolic sequence, compute the average negative log-probability of transitions.
    Returns an array of anomaly scores (one per window starting position).
    """
    scores = []
    for i in range(len(symbols) - window_size):
        window = symbols[i:i+window_size]
        score = 0.0
        valid_transitions = 0
        for j in range(len(window)-1):
            s = window[j]
            s_next = window[j+1]
            p = model.get(s, {}).get(s_next, 1e-6)
            score += -np.log(p)
            valid_transitions += 1
        scores.append(score / valid_transitions if valid_transitions > 0 else 0)
    return np.array(scores)


# Symbolic Transformation Functions for SDA and IMPACTS

def downsample(series, factor):
    """
    Downsample a time series by taking every 'factor'-th point.
    """
    return series[::factor]

def sda_transform(series, alphabet_size=5, skip_factor=4):
    """
    SDA transformation:
    """
    diff = np.diff(series)
    diff_ds = downsample(diff, skip_factor)
    breakpoints = np.percentile(diff_ds, np.linspace(0, 100, alphabet_size+1)[1:-1])
    indices = np.searchsorted(breakpoints, diff_ds, side='left')
    return "".join(chr(97 + i) for i in indices)

def impacts_transform(series, alphabet_size=5, skip_factor=4):
    """
    IMPACTS transformation:
    """
    eps = 1e-6
    ratio = series[1:] / (series[:-1] + eps)
    ratio_ds = downsample(ratio, skip_factor)
    breakpoints = np.percentile(ratio_ds, np.linspace(0, 100, alphabet_size+1)[1:-1])
    indices = np.searchsorted(breakpoints, ratio_ds, side='left')
    return "".join(chr(97 + i) for i in indices)

# Anomaly Detection Experiment (SAX, SDA, IMPACTS)
def anomaly_detection_experiment():
    # Parameters for symbolic representations
    sax_length = 256    # SAX word length
    alphabet_size = 6   # Alphabet size for SAX
    skip_factor = 4     # Downsampling factor for SDA and IMPACTS
    window_size = 6     # Sliding window for anomaly scoring

    # Generate training series (normal behavior): noisy sine wave of length 1000
    train_series = generate_noisy_sine(length=1000, freq=5, noise_std=0.1, phase=0)
    
    # SAX-based Model 
    sax_instance = SAX(word_size=sax_length, alphabet_size=alphabet_size)
    train_sax_str = sax_instance.sax_transform(train_series)
    model_sax = build_markov_model(train_sax_str)
    
    # SDA-based Model 
    train_sda_str = sda_transform(train_series, alphabet_size=5, skip_factor=skip_factor)
    model_sda = build_markov_model(train_sda_str)
    
    # IMPACTS-based Model 
    train_impacts_str = impacts_transform(train_series, alphabet_size=5, skip_factor=skip_factor)
    model_impacts = build_markov_model(train_impacts_str)
    
    # Generate test series with anomalies: same as training, but inject anomalies at positions 250, 500, and 750
    test_series = generate_noisy_sine(length=1000, freq=5, noise_std=0.1, phase=0)
    anomaly_positions = [250, 500, 750]
    test_series_anom = inject_anomalies(test_series, anomaly_positions, anomaly_width=20, freq_change=2, phase_shift=np.pi/2)
    
    # Obtain symbolic representations for the test series using each method
    test_sax_str = sax_instance.sax_transform(test_series_anom)
    test_sda_str = sda_transform(test_series_anom, alphabet_size=5, skip_factor=skip_factor)
    test_impacts_str = impacts_transform(test_series_anom, alphabet_size=5, skip_factor=skip_factor)
    
    # Compute anomaly scores using sliding windows
    scores_sax = compute_anomaly_score(test_sax_str, model_sax, window_size=window_size)
    scores_sda = compute_anomaly_score(test_sda_str, model_sda, window_size=window_size)
    scores_impacts = compute_anomaly_score(test_impacts_str, model_impacts, window_size=window_size)
    
    # Plot the test series and anomaly scores
    t = np.linspace(0, 1, 1000)
    plt.figure(figsize=(12,10))
    
    # Plot the test time series with anomalies
    plt.subplot(3,1,1)
    plt.plot(t, test_series_anom, label="Test Series with Anomalies")
    for pos in anomaly_positions:
        plt.axvline(x=pos/1000, color='red', linestyle='--', alpha=0.8)
    plt.title("Test Time Series with Injected Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot SAX anomaly scores
    plt.subplot(3,1,2)
    t_scores = np.linspace(0, 1, len(scores_sax))
    plt.plot(t_scores, scores_sax, label="SAX Anomaly Score", color='orange')
    for pos in anomaly_positions:
        plt.axvline(x=pos/1000, color='red', linestyle='--', alpha=0.8)
    plt.title("Anomaly Scores Using SAX-based Markov Model")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    
    # Plot SDA and IMPACTS anomaly scores for comparison
    plt.subplot(3,1,3)
    t_scores_sda = np.linspace(0, 1, len(scores_sda))
    t_scores_imp = np.linspace(0, 1, len(scores_impacts))
    plt.plot(t_scores_sda, scores_sda, label="SDA Anomaly Score", color='green')
    plt.plot(t_scores_imp, scores_impacts, label="IMPACTS Anomaly Score", color='blue')
    for pos in anomaly_positions:
        plt.axvline(x=pos/1000, color='red', linestyle='--', alpha=0.8)
    plt.title("Anomaly Scores Using SDA and IMPACTS")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    anomaly_detection_experiment()
