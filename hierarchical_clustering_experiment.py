
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sax import SAX 

#  SDA Transformation 
def sda_transform(series, alphabet_size=5):

    diff = np.diff(series)
    # Compute breakpoints based on quantiles of the differences
    breakpoints = np.percentile(diff, np.linspace(0, 100, alphabet_size + 1)[1:-1])
    indices = np.searchsorted(breakpoints, diff, side='left')
    # Map indices to letters 
    return "".join([chr(97 + i) for i in indices])

#  IMPACTS Transformation 
def impacts_transform(series, alphabet_size=5):

    series = np.array(series)
    eps = 1e-6  # to avoid division by zero
    ratio = series[1:] / (series[:-1] + eps)
    # Compute breakpoints using quantiles of the ratios
    breakpoints = np.percentile(ratio, np.linspace(0, 100, alphabet_size + 1)[1:-1])
    indices = np.searchsorted(breakpoints, ratio, side='left')
    return "".join([chr(97 + i) for i in indices])

#  Symbolic Distance Function (Hamming) 
def symbolic_hamming_distance(sym1, sym2):
    L = min(len(sym1), len(sym2))
    if L == 0:
        return 0.0
    diff = sum(1 for i in range(L) if sym1[i] != sym2[i])
    return diff / L

#  Data Loading Functions 
def load_synthetic_control(file_path):

    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    data = df.values.astype(float)
    return data

def subset_data(data):

    indices = [0, 10, 20, 300, 301, 302, 400, 401, 402]
    subset = data[indices, :]
    return subset, indices

#  Compute Pairwise Distance Matrix for Symbolic Representations 
def compute_distance_matrix_symbolic(sym_list, distance_func):

    N = len(sym_list)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = distance_func(sym_list[i], sym_list[j])
            D[i, j] = d
            D[j, i] = d
    return D

# Function to map indices to labels
def map_labels(indices):
    mapping = {
        **dict.fromkeys([0, 10, 20], "Normal"),
        **dict.fromkeys([300, 301, 302], "Decreasing Trend"),
        **dict.fromkeys([400, 401, 402], "Upward Shift"),
    }
    return [mapping.get(idx, f"Unknown-{idx}") for idx in indices]  # Default label if index not found


#  Plot Dendrogram 
def plot_dendrogram(D, indices, title):

    labels = map_labels(indices)
    Z = linkage(squareform(D), method='complete')
    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=labels, leaf_font_size=8)
    plt.title(title)
    plt.ylabel("Distance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#  Main Hierarchical Clustering Comparison 
def main():
    # Load the Synthetic Control dataset
    file_path = "synthetic_control_chart_time_series/synthetic_control.data"
    data = load_synthetic_control(file_path)
    
    # Extract a subset of 9 time series (from Normal, Decreasing Trend, Upward Shift)
    subset, indices = subset_data(data)
    print("Subset shape:", subset.shape)
    print("Original indices:", indices)
    
    #  Hierarchical Clustering on Raw Data (Euclidean) 
    D_raw = pdist(subset, metric='euclidean')
    Z_raw = linkage(D_raw, method='complete')
    plt.figure(figsize=(8, 4))
    dendrogram(Z_raw, labels=map_labels(indices), leaf_font_size=8)
    plt.title("Hierarchical Clustering (Raw Euclidean)")
    plt.ylabel("Distance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    #  Hierarchical Clustering on SAX Representations 
    word_size = 10      
    alphabet_size_sax = 4
    sax_instance = SAX(word_size=word_size, alphabet_size=alphabet_size_sax)
    sax_words = [sax_instance.sax_transform(ts) for ts in subset]
    print("\nSample SAX representations:")
    for i, word in enumerate(sax_words):
        print(f"Series {indices[i]}: {word}")
    
    D_sax = np.zeros((len(sax_words), len(sax_words)))
    for i in range(len(sax_words)):
        for j in range(i+1, len(sax_words)):
            d = sax_instance.mindist(sax_words[i], sax_words[j])
            D_sax[i, j] = d
            D_sax[j, i] = d
    plot_dendrogram(D_sax, indices, "Hierarchical Clustering (SAX, MINDIST)")
    
    #  Hierarchical Clustering on SDA Representations 
    alphabet_size_sda = 4
    sda_words = [sda_transform(ts, alphabet_size=alphabet_size_sda) for ts in subset]
    print("\nSample SDA representations:")
    for i, word in enumerate(sda_words):
        print(f"Series {indices[i]}: {word}")
    
    D_sda = compute_distance_matrix_symbolic(sda_words, symbolic_hamming_distance)
    plot_dendrogram(D_sda, indices, "Hierarchical Clustering (SDA, Hamming)")
    
    #  Hierarchical Clustering on IMPACTS Representations 
    alphabet_size_impacts = 4
    impacts_words = [impacts_transform(ts, alphabet_size=alphabet_size_impacts) for ts in subset]
    print("\nSample IMPACTS representations:")
    for i, word in enumerate(impacts_words):
        print(f"Series {indices[i]}: {word}")
    
    D_impacts = compute_distance_matrix_symbolic(impacts_words, symbolic_hamming_distance)
    plot_dendrogram(D_impacts, indices, "Hierarchical Clustering (IMPACTS, Hamming)")

if __name__ == "__main__":
    main()
