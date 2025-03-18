
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sax import SAX  

# Data Loading Functions

def load_control_chart(dataset_dir):

    file_path = os.path.join(dataset_dir, "synthetic_control.data")
    data = np.loadtxt(file_path)
    n_samples = data.shape[0]
    labels = np.array([i // 100 for i in range(n_samples)])
    
    train_indices = []
    test_indices = []
    for class_idx in range(6):
        start = class_idx * 100
        idx = np.arange(start, start + 100)
        train_indices.extend(idx[:50])
        test_indices.extend(idx[50:])
    X_train = data[train_indices, :]
    X_test = data[test_indices, :]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    return X_train, y_train, X_test, y_test

def load_ucr_split(dataset_dir, dataset_name):
    
    train_path = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
    df_train = pd.read_csv(train_path, header=None, delim_whitespace=True)
    df_test = pd.read_csv(test_path, header=None, delim_whitespace=True)
    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values.astype(float)
    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values.astype(float)
    return X_train, y_train, X_test, y_test

# Regression Tree (RT) Representation (APCA)
def rt_transform(series, Nmax=8):
    """
    Compute an APCA representation of a time series using a best-first segmentation strategy.
    
    This function starts with the entire series as one segment and iteratively splits the segment
    that yields the maximum reduction in total squared error (variance * length) until Nmax segments are obtained.
    
    Parameters:
        series: 1D NumPy array representing the time series.
        Nmax: desired number of segments.
    
    Returns:
        A NumPy array of length Nmax containing the mean value for each segment, in order.
    """
    n = len(series)
    # Initialize with one segment covering the entire series.
    segments = [(0, n)]
    
    # Function to compute error (sum of squared deviations) for a segment.
    def segment_error(s, e):
        seg = series[s:e]
        mean_val = np.mean(seg)
        return np.sum((seg - mean_val) ** 2)
    
    # Total error for a segment.
    seg_errors = { (s,e): segment_error(s,e) for (s,e) in segments }
    
    # While we have fewer than Nmax segments, perform the best split.
    while len(segments) < Nmax:
        best_improve = -np.inf
        best_split = None  # (segment_index, split_index)
        # Try splitting each segment at every possible split point
        for idx, (s, e) in enumerate(segments):
            if e - s < 2:
                continue  # Cannot split a segment with less than 2 points
            current_error = seg_errors[(s,e)]
            # For speed, consider splits in steps (e.g., every 2 points)
            for split in range(s+1, e):
                # Compute error for left and right segments.
                error_left = segment_error(s, split)
                error_right = segment_error(split, e)
                new_error = error_left + error_right
                improvement = current_error - new_error
                if improvement > best_improve:
                    best_improve = improvement
                    best_split = (idx, split)
        if best_split is None:
            break
        # Perform the split
        seg_idx, split_point = best_split
        s, e = segments[seg_idx]
        # Remove the segment that is split.
        segments.pop(seg_idx)
        # Remove its error record.
        seg_errors.pop((s,e))
        # Add the two new segments.
        segments.insert(seg_idx, (s, split_point))
        segments.insert(seg_idx+1, (split_point, e))
        seg_errors[(s, split_point)] = segment_error(s, split_point)
        seg_errors[(split_point, e)] = segment_error(split_point, e)
    
    # For each segment, compute the mean.
    rt_representation = np.array([np.mean(series[s:e]) for (s,e) in segments])
    return rt_representation

def rt_transform_dataset(X, Nmax=8):
    """
    Apply the RT transformation to each time series in X.
    Returns a 2D array of shape (len(X), Nmax).
    """
    return np.array([rt_transform(ts, Nmax) for ts in X])

# SAX Transformation 
def sax_transform_dataset(X, w=8, a=6):
    """
    Convert each time series in X into a SAX representation using Our SAX implementation.
    Returns a 2D array (n_samples x w) of integer codes (0 to a-1).
    """
    sax_instance = SAX(word_size=w, alphabet_size=a)
    X_sax = []
    for ts in X:
        word = sax_instance.sax_transform(ts)
        codes = [ord(ch) - ord('a') for ch in word]
        X_sax.append(codes)
    return np.array(X_sax)

# Decision Tree Classification Experiment (Multiple Runs)
def decision_tree_experiment_rt(X_train, y_train, X_test, y_test, runs=10, Nmax=8):
    """
    Run decision tree classification using RT (APCA) representation.
    Returns (mean_error, std_error).
    """
    errors = []
    X_train_rt = rt_transform_dataset(X_train, Nmax=Nmax)
    X_test_rt = rt_transform_dataset(X_test, Nmax=Nmax)
    for r in range(runs):
        clf = DecisionTreeClassifier(random_state=r)
        clf.fit(X_train_rt, y_train)
        pred = clf.predict(X_test_rt)
        err = np.mean(pred != y_test)
        errors.append(err)
    return np.mean(errors), np.std(errors)

def decision_tree_experiment_sax(X_train, y_train, X_test, y_test, runs=10, w=8, a=6):
    """
    Run decision tree classification using SAX representation.
    Returns (mean_error, std_error).
    """
    errors = []
    X_train_sax = sax_transform_dataset(X_train, w=w, a=a)
    X_test_sax = sax_transform_dataset(X_test, w=w, a=a)
    for r in range(runs):
        clf = DecisionTreeClassifier(random_state=r)
        clf.fit(X_train_sax, y_train)
        pred = clf.predict(X_test_sax)
        err = np.mean(pred != y_test)
        errors.append(err)
    return np.mean(errors), np.std(errors)

# Main Experiment 
def run_decision_tree_experiment(dataset_name, dataset_dir):
    """
    Run decision tree classification experiment on a given dataset.
    Returns a dictionary with error rates (mean ± std) for RT and SAX.
    """
    if dataset_name.lower() in ["controlchart", "control_chart"]:
        X_train, y_train, X_test, y_test = load_control_chart(dataset_dir)
    else:
        X_train, y_train, X_test, y_test = load_ucr_split(dataset_dir, dataset_name)
    
    print(f"\nDataset {dataset_name}:")
    print(f"Training: {X_train.shape[0]} instances; Test: {X_test.shape[0]} instances; Series length: {X_train.shape[1]}")
    
    rt_mean, rt_std = decision_tree_experiment_rt(X_train, y_train, X_test, y_test, runs=10, Nmax=8)
    sax_mean, sax_std = decision_tree_experiment_sax(X_train, y_train, X_test, y_test, runs=10, w=8, a=6)
    
    print(f"\nDecision Tree Results for {dataset_name}:")
    print(f"RT (APCA) error: {rt_mean:.2f} ± {rt_std:.2f}")
    print(f"SAX error: {sax_mean:.2f} ± {sax_std:.2f}")
    
    return {"rt_mean": rt_mean, "rt_std": rt_std, "sax_mean": sax_mean, "sax_std": sax_std}

def load_ucr_split(dataset_dir, dataset_name):

    train_path = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
    df_train = pd.read_csv(train_path, header=None, delim_whitespace=True)
    df_test = pd.read_csv(test_path, header=None, delim_whitespace=True)
    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values.astype(float)
    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values.astype(float)
    return X_train, y_train, X_test, y_test

def main():
    np.random.seed(42)
    random.seed(42)
    
    dataset_dir_cc = "synthetic_control_chart_time_series"  
    dataset_dir_cbf = "UCRArchive_2018/CBF"  
    
    print("***** Control Chart Dataset *****")
    results_cc = run_decision_tree_experiment("ControlChart", dataset_dir_cc)
    
    print("\n***** Cylinder-Bell-Funnel (CBF) Dataset *****")
    results_cbf = run_decision_tree_experiment("CBF", dataset_dir_cbf)
    
    # Plot results
    labels = ["Control Chart", "CBF"]
    rt_errors = [results_cc["rt_mean"]*100, results_cbf["rt_mean"]*100]
    sax_errors = [results_cc["sax_mean"]*100, results_cbf["sax_mean"]*100]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6,4))
    rects1 = ax.bar(x - width/2, rt_errors, width, yerr=[results_cc["rt_std"]*100, results_cbf["rt_std"]*100],
                     label="RT (APCA)", capsize=5)
    rects2 = ax.bar(x + width/2, sax_errors, width, yerr=[results_cc["sax_std"]*100, results_cbf["sax_std"]*100],
                     label="SAX", capsize=5)
    
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Decision Tree Classification Error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
