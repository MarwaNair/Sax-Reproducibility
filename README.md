
# **Reproducibility of SAX Paper Experiments**

This repository contains the implementation of experiments reproducing the results from the paper:  
**"A Symbolic Representation of Time Series, with Implications for Streaming Algorithms"** by **Lin et al. (2003)**.

## **Project Overview**
The objective of this project is to assess the reproducibility of the experimental evaluation presented in the original SAX paper. The experiments cover key tasks in time series data mining, including clustering and classification. This repository provides the full implementation for replicating these experiments, analyzing results, and comparing SAX to other time series distance measures.

## **Repository Structure**
The repository is structured as follows:
```
├── decision_tree_experiment.py       # Decision tree classification experiment 
├── hierarchical_clustering_experiment.py  # Hierarchical clustering experiment 
├── kmeans_experiment.py              # K-means clustering experiment 
├── nn_classification_experiment.py   # 1-NN classification experiment 
├── sax.py                            # SAX implementation 
├── test_sax.py                       # Testuing SAX implementation with the pyts library implementation 
├── synthetic+control+chart+time+series.zip  # Dataset files for the Control Chart experiments
├── README.md                         # This documentation file
```

---

## **Implemented Experiments**
Each script corresponds to a different experiment conducted in the original paper:

### **1. K-Means Clustering Experiment (`kmeans_experiment.py`)**
- Compares k-means clustering performance on raw time series vs. SAX representation.
- Evaluates clustering over iterations using the **Within-Cluster Sum of Squares (WCSS)** objective function.
- Computes **Silhouette Score** to assess clustering quality.

### **2. Hierarchical Clustering Experiment (`hierarchical_clustering_experiment.py`)**
- Performs hierarchical clustering.
- Compare SAX’s MINDIST distance to Euclidean Distance, SDA and IMPACTS for Hierarchical Clustering.
- Generates dendrograms to visualize clustering results.

### **3. Nearest-Neighbor Classification Experiment (`nn_classification_experiment.py`)**
- Implements **1-Nearest Neighbor (1-NN) classification**.
- Compares SAX with:
  - Euclidean Distance
  - IMPACTS
  - SDA
  - \( L_{\infty} \) norm
- Evaluates classification error rate across varying alphabet sizes.

### **4. Decision Tree Classification Experiment (`decision_tree_experiment.py`)**
- Compares SAX-based classification with the **Regression Tree (RT) method**.
- Uses SAX representations as features and evaluates decision tree accuracy.
- Implements the **piecewise constant approximation** technique for RT-based classification.

---

## **How to Run the Experiments**
### **1. Setup the Environment**
Ensure you have Python 3.x installed along with the required dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### **2. Running Individual Experiments**
Each experiment can be run independently. For example:

- **Run k-means clustering experiment:**
  ```bash
  python kmeans_experiment.py
  ```

- **Run hierarchical clustering experiment:**
  ```bash
  python hierarchical_clustering_experiment.py
  ```

- **Run 1-NN classification experiment:**
  ```bash
  python nn_classification_experiment.py
  ```

- **Run decision tree classification experiment:**
  ```bash
  python decision_tree_experiment.py
  ```

---

## **Datasets**
This repository includes the dataset **Synthetic Control Chart Time Series** (`synthetic_control.data`).  
Additional datasets (e.g., CBF dataset in the UCR archive) should be downloaded automatically when running the code.

---

## **Results and Observations**
The results obtained from these experiments were compared to those reported in the original paper:

- **Hierarchical Clustering performance:** The SAX-based hierarchical clustering **correctly grouped classes into distinct subtrees** while the ED, SDA and IMPACTS  failed to separate noisy classes correctly, as observed in the dendrogram. The smoothing effect of SAX improved clustering quality, reducing the impact of time-series shifts and noise.
- **Partial Clustering performance:** SAX-based k-means clustering showed compact clusters with a slightly higher objective function compared to Euclidean distance (but a better Silhoutte score).
- **Classification accuracy:** SAX achieved competitive results against Euclidean distance and outperformed other symbolic representations like IMPACTS and SDA.
- **Decision trees:** SAX-based decision trees were comparable in performance to the Regression Tree approach.

---

## **Contributors**
- **Marwa Nair**  
  Master IASD, Université Paris-Dauphine  
  NoSQL Course Project (2025)  

---
