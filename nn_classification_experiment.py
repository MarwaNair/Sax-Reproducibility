import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sax import SAX  
from sklearn.neighbors import KNeighborsClassifier


# Loading datasets functions

def load_cbf_data(dataset_dir):
    """
    Loads the CBF dataset from two files:
      - CBF_TRAIN.tsv
      - CBF_TEST.tsv
    Returns (X_train, y_train, X_test, y_test).
    """
    train_path = os.path.join(dataset_dir, "CBF_TRAIN.tsv")
    test_path = os.path.join(dataset_dir, "CBF_TEST.tsv")
    
    def load_tsv(path):
        data = []
        labels = []
        with open(path, "r") as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                data.append(list(map(float, vals[1:])))
        return np.array(data), np.array(labels)
    
    X_train, y_train = load_tsv(train_path)
    X_test, y_test = load_tsv(test_path)
    return X_train, y_train, X_test, y_test

def load_control_chart(dataset_dir):
    """
    Loads the Control Chart dataset from 'synthetic_control.data'.
    Splits each class (100 examples) into 50 train / 50 test.
    Returns (X_train, y_train, X_test, y_test).
    """
    path = os.path.join(dataset_dir, "synthetic_control.data")
    raw_data = np.loadtxt(path)
    n_samples, series_len = raw_data.shape
    
    train_indices = []
    test_indices = []
    labels = np.array([i // 100 for i in range(n_samples)])  # 6 classes, each 100
    
    for class_idx in range(6):
        start = class_idx * 100
        idx = np.arange(start, start + 100)
        train_indices.extend(idx[:50])
        test_indices.extend(idx[50:])
    
    X_train = raw_data[train_indices]
    y_train = labels[train_indices]
    X_test = raw_data[test_indices]
    y_test = labels[test_indices]
    return X_train, y_train, X_test, y_test


#  Classification Utilities

def nn_test_euclidean(X_train, y_train, X_test, y_test):
    """
    1-NN with Euclidean distance on raw data.
    Returns error rate.
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return np.mean(pred != y_test)

def nn_test_lpinf(X_train, y_train, X_test, y_test):
    """
    1-NN with Chebyshev (L-infinity) distance on raw data.
    Returns error rate.
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric='chebyshev')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return np.mean(pred != y_test)

def normalized_hamming(s1, s2):
    """
    Normalized Hamming distance between two symbolic strings.
    """
    L = min(len(s1), len(s2))
    if L == 0:
        return 0.0
    return sum(s1[i] != s2[i] for i in range(L)) / L

def nn_test_symbolic(X_train, y_train, X_test, y_test, transform_func, **kwargs):
    """
    1-NN for symbolic approaches (SDA, IMPACTS).
    transform_func: function(series, **kwargs) -> symbolic string
    kwargs: parameters for transform_func
    """
    train_sym = [transform_func(ts, **kwargs) for ts in X_train]
    test_sym = [transform_func(ts, **kwargs) for ts in X_test]
    
    errors = 0
    for i, sym_test in enumerate(test_sym):
        best_dist = float('inf')
        best_label = None
        for j, sym_train in enumerate(train_sym):
            d = normalized_hamming(sym_test, sym_train)
            if d < best_dist:
                best_dist = d
                best_label = y_train[j]
        if best_label != y_test[i]:
            errors += 1
    return errors / len(X_test)

def nn_test_sax(X_train, y_train, X_test, y_test, w, a):
    """
    1-NN classification using SAX with parameters (w, a).
    """
    sax_instance = SAX(word_size=w, alphabet_size=a)
    train_sax = [sax_instance.sax_transform(ts) for ts in X_train]
    test_sax = [sax_instance.sax_transform(ts) for ts in X_test]
    
    errors = 0
    for i, sym_test in enumerate(test_sax):
        best_dist = float('inf')
        best_label = None
        for j, sym_train in enumerate(train_sax):
            d = sax_instance.mindist(sym_test, sym_train)
            if d < best_dist:
                best_dist = d
                best_label = y_train[j]
        if best_label != y_test[i]:
            errors += 1
    return errors / len(X_test)

#  SDA & IMPACTS vanilla implementations

def downsample(series, factor):
    return series[::factor]

def sda_transform(series, alphabet_size=5, skip_factor=4):
    diff = np.diff(series)
    diff_ds = downsample(diff, skip_factor)
    breakpoints = np.percentile(diff_ds, np.linspace(0, 100, alphabet_size+1)[1:-1])
    indices = np.searchsorted(breakpoints, diff_ds)
    return "".join(chr(97 + i) for i in indices)

def impacts_transform(series, alphabet_size=5, skip_factor=4):
    eps = 1e-6
    ratio = series[1:] / (series[:-1] + eps)
    ratio_ds = downsample(ratio, skip_factor)
    breakpoints = np.percentile(ratio_ds, np.linspace(0, 100, alphabet_size+1)[1:-1])
    indices = np.searchsorted(breakpoints, ratio_ds)
    return "".join(chr(97 + i) for i in indices)

#  Main Experiment Code

def experiment_alphabet_sizes(X_train, y_train, X_test, y_test):
    """
    For a range of alphabet sizes from 3 to 10:
      - compute error rates for:
        Euclidean (fixed, no alphabet)
        LPinf (fixed, no alphabet)
        SDA (alphabet fixed = 5 => single error)
        IMPACTS (varies with a)
        SAX (varies with a)
    Return dict of lists: 
      "eucl", "lpinf", "sda", "impacts", "sax"
    each is a list of length (10-3+1) = 8, except sda, eucl, lpinf are repeated lines.
    """
    alph_range = range(3, 11)  
    n_alph = len(list(alph_range))
    
    # Precompute Euclidean & LPinf once
    err_eucl = nn_test_euclidean(X_train, y_train, X_test, y_test)
    err_lpinf = nn_test_lpinf(X_train, y_train, X_test, y_test)
    
    # Precompute SDA once (alphabet fixed at 5 => skip_factor=4)
    err_sda = nn_test_symbolic(X_train, y_train, X_test, y_test, 
                               transform_func=sda_transform, 
                               alphabet_size=5, skip_factor=4)
    
    # Build arrays of repeated values for plotting
    eucl_list = [err_eucl]*n_alph
    lpinf_list = [err_lpinf]*n_alph
    sda_list = [err_sda]*n_alph
    
    # For IMPACTS & SAX, vary the alphabet size
    impacts_list = []
    sax_list = []
    for a in alph_range:
        # IMPACTS
        err_imp = nn_test_symbolic(X_train, y_train, X_test, y_test,
                                   transform_func=impacts_transform,
                                   alphabet_size=a, skip_factor=4)
        impacts_list.append(err_imp)
        
        # For SAX, we need to pick a dimensionality reduction approach
        # In the paper, they mention "dimensionality reduction of 4 to 1"
        # meaning if the series length is n, we reduce to n/4 segments
        
        w = 16
        err_sax = nn_test_sax(X_train, y_train, X_test, y_test, w, a)
        sax_list.append(err_sax)
    
    return {
        "alphabet_sizes": list(alph_range),
        "eucl": eucl_list,
        "lpinf": lpinf_list,
        "sda": sda_list,
        "impacts": impacts_list,
        "sax": sax_list
    }

def plot_comparison(results_cbf, results_cc):
    """
    Plot two subplots each showing error rates vs. alphabet size
    for the 5 methods: IMPACTS, SDA, Euclidean, LPinf, SAX.
    """
    alph_sizes = results_cbf["alphabet_sizes"] 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # CBF
    ax1 = axes[0]
    ax1.plot(alph_sizes, results_cbf["impacts"], label="IMPACTS", linestyle="--", color="blue")
    ax1.plot(alph_sizes, results_cbf["sda"], label="SDA", linestyle="--", color="purple")
    ax1.plot(alph_sizes, results_cbf["eucl"], label="Euclidean", linestyle=":", color="red")
    ax1.plot(alph_sizes, results_cbf["lpinf"], label="LP∞", linestyle=":", color="orange")
    ax1.plot(alph_sizes, results_cbf["sax"], label="SAX", linestyle="-", color="brown")
    
    ax1.set_title("Cylinder-Bell-Funnel")
    ax1.set_xlabel("Alphabet Size")
    ax1.set_ylabel("Error Rate")
    ax1.set_xticks(alph_sizes)
    ax1.set_ylim([0,1])  
    ax1.legend(loc="best")
    
    # CC
    ax2 = axes[1]
    ax2.plot(alph_sizes, results_cc["impacts"], label="IMPACTS", linestyle="--", color="blue")
    ax2.plot(alph_sizes, results_cc["sda"], label="SDA", linestyle="--", color="purple")
    ax2.plot(alph_sizes, results_cc["eucl"], label="Euclidean", linestyle=":", color="red")
    ax2.plot(alph_sizes, results_cc["lpinf"], label="LP∞", linestyle=":", color="orange")
    ax2.plot(alph_sizes, results_cc["sax"], label="SAX", linestyle="-", color="brown")
    
    ax2.set_title("Control Chart")
    ax2.set_xlabel("Alphabet Size")
    ax2.set_xticks(alph_sizes)
    ax2.set_ylim([0,1])  
    
    plt.tight_layout()
    plt.show()


#  Main

def main():
    np.random.seed(42)
    random.seed(42)
    
    dataset_dir_cbf = "UCRArchive_2018/CBF"
    dataset_dir_cc = "synthetic_control_chart_time_series"
    
    # --- Load CBF data ---
    X_train_cbf, y_train_cbf, X_test_cbf, y_test_cbf = load_cbf_data(dataset_dir_cbf)
    # Build results
    results_cbf = experiment_alphabet_sizes(X_train_cbf, y_train_cbf, X_test_cbf, y_test_cbf)
    
    # --- Load Control Chart data ---
    X_train_cc, y_train_cc, X_test_cc, y_test_cc = load_control_chart(dataset_dir_cc)
    # Build results
    results_cc = experiment_alphabet_sizes(X_train_cc, y_train_cc, X_test_cc, y_test_cc)
    
    # Plot
    plot_comparison(results_cbf, results_cc)

if __name__ == "__main__":
    main()
