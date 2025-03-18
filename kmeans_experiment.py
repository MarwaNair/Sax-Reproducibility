
import numpy as np
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sax import SAX  
import os
os.environ["OMP_NUM_THREADS"] = "2"

def plot_objective_function(obj_history_raw, obj_history_sax, title):
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(obj_history_raw) + 1), obj_history_raw, label='Raw Data k-means', marker='o')
    plt.plot(range(1, len(obj_history_sax) + 1), obj_history_sax, label='SAX k-means', marker='s')
    plt.xticks(range(1, max(len(obj_history_raw), len(obj_history_sax)) + 1))
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function (WCSS)')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')
    plt.show()
    
# Raw Data k-means (using Euclidean distance) 
def custom_kmeans_raw(raw_data, k, max_iter=100):
    """
     k-means clustering on raw data using Euclidean distance.
    
    Returns:
        centroids: NumPy array of shape (k, series_length) of final centroids
        assignments: NumPy array of cluster assignments for each sample
        iterations: number of iterations until convergence
        obj_history: list of objective function (WCSS) values at each iteration
    """
    n, d = raw_data.shape
    init_indices = random.sample(range(n), k)
    centroids = raw_data[init_indices].copy()
    assignments = np.zeros(n, dtype=int)
    obj_history = []

    for iteration in range(max_iter):
        new_assignments = np.array([np.argmin([np.linalg.norm(raw_data[i] - centroids[j])
                                               for j in range(k)]) for i in range(n)])
        # Compute objective function (WCSS)
        obj = 0.0
        for j in range(k):
            cluster_points = raw_data[new_assignments == j]
            if cluster_points.size > 0:
                obj += np.sum(np.linalg.norm(cluster_points - centroids[j], axis=1) ** 2)
        obj_history.append(obj)
        
        if np.all(new_assignments == assignments):
            return centroids, new_assignments, iteration + 1, obj_history
        assignments = new_assignments
        
        for j in range(k):
            cluster_points = raw_data[assignments == j]
            if cluster_points.size > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
    return centroids, assignments, max_iter, obj_history




# SAX k-means 
def majority_vote(cluster_words, word_size):
    """
    Compute a new centroid SAX word by taking a majority vote at each character position.
    In case of ties, choose the lexicographically smallest symbol.
    """
    if not cluster_words:
        return None
    new_centroid = []
    for i in range(word_size):
        symbols = [word[i] for word in cluster_words]
        counts = Counter(symbols)
        most_common = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        new_centroid.append(most_common)
    return "".join(new_centroid)

def custom_kmeans_sax(sax_words, raw_data, k, sax_instance, max_iter=100):
    """
    k-means clustering on SAX representations (using symbolic clustering),
    but the objective function is computed in the raw space.
    
    Returns:
        centroids: list of final SAX centroids (strings)
        assignments: NumPy array of cluster assignments for each sample
        iterations: number of iterations until convergence
        obj_history: list of objective function (WCSS) values (computed in raw space) per iteration
    """
    n = len(sax_words)
    word_size = sax_instance.word_size
    centroids = random.sample(sax_words, k)
    assignments = np.zeros(n, dtype=int)
    obj_history = []

    for iteration in range(max_iter):
        new_assignments = []
        for word in sax_words:
            distances = [sax_instance.mindist(word, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            new_assignments.append(cluster_idx)
        new_assignments = np.array(new_assignments)
        
        # Compute raw centroids for current clusters and WCSS in raw space.
        obj = 0.0
        for j in range(k):
            indices = np.where(new_assignments == j)[0]
            if len(indices) > 0:
                centroid_raw = np.mean(raw_data[indices], axis=0)
            else:
                centroid_raw = raw_data[random.choice(range(n))]
            for idx in indices:
                obj += np.linalg.norm(raw_data[idx] - centroid_raw) ** 2
        obj_history.append(obj)
        
        if np.all(new_assignments == assignments):
            return centroids, new_assignments, iteration + 1, obj_history
        assignments = new_assignments
        
        # Update SAX centroids using majority vote in symbolic space.
        new_centroids = []
        for j in range(k):
            cluster_words = [sax_words[i] for i in range(n) if assignments[i] == j]
            centroid_sax = majority_vote(cluster_words, word_size) if cluster_words else random.choice(sax_words)
            new_centroids.append(centroid_sax)
        centroids = new_centroids

    return centroids, assignments, max_iter, obj_history

# Function to load a UCR dataset file 
def load_ucr_dataset(file_path):
    """
    Load a UCR dataset file.
    Assumes the file is whitespace-delimited, with the first column as the label,
    and the remaining columns as time series values.
    Returns: (labels, data) where data is a NumPy array of shape (n_samples, n_timestamps)
    """
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    labels = df.iloc[:, 0].values
    data = df.iloc[:, 1:].values.astype(float)
    return labels, data

# Function to simulate a Space Shuttle telemetry dataset
def generate_simulated_spaceshuttle(n_samples=1000, series_length=512):
    """
    Generate a simulated Space Shuttle telemetry dataset.
    
    We simulate three clusters:
      - Cluster 1: Increasing trend with two periodic components and occasional plateaus.
      - Cluster 2: Decreasing trend with periodic oscillations and sudden random spikes.
      - Cluster 3: Stable signal with low noise and occasional transient anomalies.
      
    Returns:
        true_labels: NumPy array of shape (n_samples,) with cluster labels (0, 1, or 2).
        data: NumPy array of shape (n_samples, series_length) containing the simulated time series.
    """
    data = []
    labels = []
    t = np.linspace(0, 1, series_length)
    
    n_clusters = 3
    base = n_samples // n_clusters
    extras = n_samples % n_clusters
    counts = [base + (1 if i < extras else 0) for i in range(n_clusters)]
    
    # Cluster 1
    for _ in range(counts[0]):
        trend = 100 + 100 * t
        periodic = 10 * np.sin(2 * np.pi * 5 * t) + 5 * np.sin(2 * np.pi * 10 * t)
        series = trend + periodic + np.random.randn(series_length) * 10
        plateau_start = np.random.randint(0, series_length - 50)
        series[plateau_start:plateau_start+20] = np.mean(series[plateau_start:plateau_start+20])
        data.append(series)
        labels.append(0)
    
    # Cluster 2
    for _ in range(counts[1]):
        trend = 200 - 100 * t
        periodic = 10 * np.sin(2 * np.pi * 6 * t + np.pi/6)
        series = trend + periodic + np.random.randn(series_length) * 10
        spike_positions = np.random.choice(series_length, size=5, replace=False)
        for pos in spike_positions:
            series[pos] += np.random.choice([30, -30])
        data.append(series)
        labels.append(1)
    
    # Cluster 3
    for _ in range(counts[2]):
        series = np.ones(series_length) * 150 + np.random.randn(series_length) * 2
        anomaly_start = np.random.randint(0, series_length - 30)
        series[anomaly_start:anomaly_start+10] += np.random.randn(10) * 10
        data.append(series)
        labels.append(2)
    
    data = np.array(data)
    labels = np.array(labels)
    return labels, data


    
    



# Experiment on Real Dataset
def experiment_real_dataset():
    
    print("----- Real Dataset Experiment: CBF -----")
    dataset_path = "UCRArchive_2018/CBF/CBF_TEST.tsv"  
    true_labels, raw_data = load_ucr_dataset(dataset_path)
    n_samples, series_length = raw_data.shape
    print(f"Loaded {n_samples} time series, each of length {series_length}.")

    # Raw Data k-means 
    k = 3  # Number of clusters
    centroids_raw, assignments_raw, iterations_raw, obj_history_raw = custom_kmeans_raw(raw_data, k, max_iter=100)
    
    raw_silhouette = silhouette_score(raw_data, assignments_raw, metric='euclidean')
    print("\n[Custom Raw Data k-means]")
    print(f"Converged after {iterations_raw} iterations.")
    print(f"Final Objective Function (WCSS, raw): {obj_history_raw[-1]:.4f}")
    print(f"Silhouette Score (raw): {raw_silhouette:.4f}")

    # SAX k-means 
    word_size = 128  
    alphabet_size = 16
    sax_instance = SAX(word_size=word_size, alphabet_size=alphabet_size)
    sax_words = [sax_instance.sax_transform(ts) for ts in raw_data]
    
    centroids_sax, assignments_sax, sax_iterations, obj_history_sax = custom_kmeans_sax(sax_words, raw_data, k, sax_instance, max_iter=100)
    sax_silhouette = silhouette_score(raw_data, assignments_sax, metric='euclidean')
    
    print("\n[Custom SAX k-means]")
    print(f"Converged after {sax_iterations} iterations.")
    print(f"Final Objective Function (WCSS, SAX): {obj_history_sax[-1]:.4f}")
    print(f"Silhouette Score (SAX): {sax_silhouette:.4f}")
    
    # Print objective function history for comparison:
    print("\nObjective Function History (Raw Data k-means):")
    print(obj_history_raw)
    print("\nObjective Function History (SAX k-means):")
    print(obj_history_sax)
    plot_objective_function(obj_history_raw, obj_history_sax, "Objective Function Convergence : CBF Dataset")


# Experiment on Simulated Dataset
def experiment_simulated_dataset():
    print("\n----- Simulated Space Shuttle Telemetry Experiment -----")
    true_labels, raw_data = generate_simulated_spaceshuttle(n_samples=1000, series_length=512)
    n_samples, series_length = raw_data.shape
    print(f"Generated {n_samples} simulated time series, each of length {series_length}.")

    # ---------- Raw Data k-means ----------
    k = 3  # Number of clusters
    centroids_raw, assignments_raw, iterations_raw, obj_history_raw = custom_kmeans_raw(raw_data, k, max_iter=100)
    raw_silhouette = silhouette_score(raw_data, assignments_raw, metric='euclidean')
    ari_raw = adjusted_rand_score(true_labels, assignments_raw)
    nmi_raw = normalized_mutual_info_score(true_labels, assignments_raw)
    
    print("\n[Custom Raw Data k-means]")
    print(f"Converged after {iterations_raw} iterations.")
    print(f"Final Objective Function (WCSS, raw): {obj_history_raw[-1]:.4f}")
    print(f"Silhouette Score (raw): {raw_silhouette:.4f}")
    print(f"ARI (raw): {ari_raw:.4f}")
    print(f"NMI (raw): {nmi_raw:.4f}")

    # ---------- SAX k-means ----------
    word_size = 64
    alphabet_size = 8
    sax_instance = SAX(word_size=word_size, alphabet_size=alphabet_size)
    sax_words = [sax_instance.sax_transform(ts) for ts in raw_data]
    centroids_sax, assignments_sax, sax_iterations, obj_history_sax = custom_kmeans_sax(sax_words, raw_data, k, sax_instance, max_iter=100)
    sax_silhouette = silhouette_score(raw_data, assignments_sax, metric='euclidean')
    ari_sax = adjusted_rand_score(true_labels, assignments_sax)
    nmi_sax = normalized_mutual_info_score(true_labels, assignments_sax)
    
    print("\n[Custom SAX k-means]")
    print(f"Converged after {sax_iterations} iterations.")
    print(f"Final Objective Function (WCSS, SAX): {obj_history_sax[-1]:.4f}")
    print(f"Silhouette Score (SAX): {sax_silhouette:.4f}")
    print(f"ARI (SAX): {ari_sax:.4f}")
    print(f"NMI (SAX): {nmi_sax:.4f}")
    
    print("\nObjective Function History (Raw Data k-means):")
    print(obj_history_raw)
    print("\nObjective Function History (SAX k-means):")
    print(obj_history_sax)
    plot_objective_function(obj_history_raw, obj_history_sax, "Objective Function Convergence (Simulated Space Shuttle Data)")






# Main 
def main():
    
    np.random.seed(1)
    random.seed(1)
    experiment_real_dataset()
    experiment_simulated_dataset()

if __name__ == "__main__":
    main()
