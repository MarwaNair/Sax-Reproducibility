# Reproducibility of SAX Paper Experiments

This repository contains the implementation of experiments reproducing the results from the paper:  
**"A Symbolic Representation of Time Series, with Implications for Streaming Algorithms"** by Lin et al. (2003).

## Project Overview
The objective of this project is to assess the reproducibility of the experimental evaluation presented in the original SAX paper. The experiments cover key tasks in time series data mining, including clustering, classification and anomaly detection. This repository provides the full implementation for replicating these experiments, analyzing results, and comparing SAX to other time series distance measures.

## Repository Structure
```
├── decision_tree_experiment.py       # Decision tree classification experiment (RT vs. SAX)
├── hierarchical_clustering_experiment.py  # Hierarchical clustering experiment (SAX, ED, SDA, IMPACTS)
├── kmeans_experiment.py              # K-means clustering experiment (objective function and silhouette scores)
├── nn_classification_experiment.py   # 1-NN classification experiment (comparison among multiple distance measures)
├── anomaly_detection_experiment.py   # Anomaly detection experiment using a SAX-based Markov model, SDA, and IMPACTS
├── sax.py                            # SAX implementation
├── test_sax.py                       # Testing SAX implementation against pyts library version
├── synthetic+control+chart+time+series.zip  # Dataset files for the Control Chart experiments
├── README.md                         # This documentation file
```

## Implemented Experiments
Each script corresponds to a different experiment from the original paper:

### 1. K-Means Clustering Experiment (`kmeans_experiment.py`)
- Compares k-means clustering performance on raw time series vs. SAX representation.
- Evaluates clustering using the Within-Cluster Sum of Squares (WCSS) objective function and Silhouette Score.
- Tracks convergence behavior over iterations.

### 2. Hierarchical Clustering Experiment (`hierarchical_clustering_experiment.py`)
- Performs hierarchical clustering using complete linkage.
- Compares SAX’s MINDIST to Euclidean, SDA, and IMPACTS distances.
- Generates dendrograms to visualize the natural groupings.

### 3. Nearest-Neighbor Classification Experiment (`nn_classification_experiment.py`)
- Implements 1-Nearest Neighbor (1-NN) classification.
- Compares SAX with Euclidean distance, IMPACTS, SDA, and \( L_{\infty} \) norm.
- Evaluates classification error rate across varying alphabet sizes.

### 4. Decision Tree Classification Experiment (`decision_tree_experiment.py`)
- Compares decision tree classification using the Regression Tree (RT) representation (via a regression tree–based segmentation approach for APCA) with SAX-based classification.
- Runs multiple experiments to report mean error rates ± standard deviation.

### 5. Anomaly Detection Experiment (`anomaly_detection_experiment.py`)
- Implements anomaly detection by leveraging the discrete nature of symbolic representations.
- Uses a SAX-based Markov model to detect anomalous regions in a test time series with injected anomalies.
- For comparison, similar anomaly detection is performed using SDA and IMPACTS representations.
- The anomaly score is computed via a sliding window as the average negative log-probability of symbol transitions.
- Results are visualized by plotting both the test series and the computed anomaly scores, with anomalies clearly marked.

## How to Run the Experiments
### 1. Setup the Environment
Ensure you have Python 3.x installed along with the required dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### 2. Running Individual Experiments
Each experiment can be run independently. For example:
- **K-Means Clustering Experiment:**
  ```bash
  python kmeans_experiment.py
  ```
- **Hierarchical Clustering Experiment:**
  ```bash
  python hierarchical_clustering_experiment.py
  ```
- **1-NN Classification Experiment:**
  ```bash
  python nn_classification_experiment.py
  ```
- **Decision Tree Classification Experiment:**
  ```bash
  python decision_tree_experiment.py
  ```
- **Anomaly Detection Experiment:**
  ```bash
  python anomaly_detection_experiment.py
  ```

## **Datasets**
This repository includes the dataset **Synthetic Control Chart Time Series** (synthetic_control.data).  
Additional datasets (e.g., CBF dataset in the UCR archive) should be downloaded automatically when running the code.

## Results and Observations
Our experimental results are summarized as follows:
- **K-Means Clustering**: SAX-based clustering produced compact clusters with a favorable silhouette score, albeit with a slightly higher WCSS compared to raw Euclidean clustering, indicating a smoothing effect.
- **Hierarchical Clustering**: Dendrograms showed that SAX (using MINDIST) was better at grouping similar time series than Euclidean distance, SDA, and IMPACTS—especially in noisy scenarios.
- **Nearest-Neighbor Classification**: SAX-based 1-NN classification achieved competitive error rates compared to raw Euclidean distance and outperformed other symbolic methods like SDA and IMPACTS.
- **Decision Tree Classification**: When comparing the RT (Regression Tree) approach based on a proper APCA segmentation with the SAX representation, SAX demonstrated competitive classification error with the advantage of being more scalable.
- **Anomaly Detection**: The SAX-based Markov model clearly identified anomalous segments in the test series with minimal false alarms, whereas SDA and IMPACTS didn't.

## Contributors
- **Marwa Nair**  
  Master IASD, Université Paris-Dauphine  
  NoSQL Course Project (2025)

## License
This project is released under the MIT License.

