import numpy as np
import pandas as pd
import requests
import zipfile
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from sax import SAX  # Import our SAX implementation



def z_normalize(series):
        """Normalize the time series to have zero mean and unit variance."""
        mean, std = np.mean(series), np.std(series)
        return (series - mean) / std if std > 0 else np.zeros_like(series)
    
    
    
# Download and Extract Coffee dataset
ucr_url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
dataset_name = "Coffee"

# Download only if not already present
if not os.path.exists("UCRArchive_2018.zip"):
    print("Downloading UCR dataset...")
    response = requests.get(ucr_url, stream=True)
    with open("UCRArchive_2018.zip", "wb") as f:
        f.write(response.content)

# Extract Coffee dataset
if not os.path.exists(f"UCRArchive_2018/{dataset_name}"):
    with zipfile.ZipFile("UCRArchive_2018.zip", "r") as zip_ref:
        zip_ref.extractall("UCRArchive_2018")

# Load Coffee dataset
train_file = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TRAIN.tsv"
test_file = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TEST.tsv"

train_data = pd.read_csv(train_file, sep="\t", header=None)
test_data = pd.read_csv(test_file, sep="\t", header=None)

# Combine train and test sets
data = pd.concat([train_data, test_data], axis=0)

# Extract labels and time series
labels = data.iloc[:, 0].values  # First column is class label
time_series = data.iloc[:, 1:].values  # Rest is the time series data

# Select one example time series 
ts_example = time_series[0]

# Apply Our SAX Implementation
alphabet_size = 4
word_size = 16
sax = SAX(word_size=word_size, alphabet_size=alphabet_size)  # Define SAX parameters
our_sax_word = sax.sax_transform(ts_example)  # Convert time series to SAX

# Apply pyts SAX Implementation for Comparison

# # Step 1: Normalize the time series (z-normalization)
ts_example_norm = z_normalize(ts_example)

# # Step 2: Compute PAA (reduce the dimensionality)
paa = PiecewiseAggregateApproximation(window_size=None, output_size=word_size)
ts_example_paa = paa.transform(ts_example_norm.reshape(1, -1))[0]

# # Step 3: Apply SAX on the PAA-transformed series
n_bins = alphabet_size
pyts_sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy="normal")
pyts_sax_word = pyts_sax.transform(ts_example_paa.reshape(1, -1))[0]

# Step 4: Compare Results
print(f"Our SAX Word:      {our_sax_word}")
print(f"pyts SAX Word:      {''.join(pyts_sax_word)}")
print(f"Are SAX words equal? {our_sax_word == ''.join(pyts_sax_word)}")

if not (our_sax_word == ''.join(pyts_sax_word)):
    print("SAX words are not equal")
    print(f"The difference is at positions {[(i, our_sax_word[i], pyts_sax_word[i]) for i in range(len(our_sax_word)) if our_sax_word[i] != pyts_sax_word[i]]}")



# Plot Time Series with SAX Segmentation

# Compute PAA values using Our SAX object's PAA transform
paa_values = sax.paa_transform(ts_example)

# Compute segmentation boundaries and the segment centers
n = len(ts_example)
bounds = np.linspace(0, n, word_size + 1, dtype=int)
centers = [(bounds[i] + bounds[i+1]) / 2 for i in range(word_size)]

# Compute Gaussian breakpoints used in SAX transformation
# For alphabet_size=4, we have 3 breakpoints.
bins = stats.norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

# Create the figure
plt.figure(figsize=(12, 6))
plt.plot(ts_example, 'k-', label="Original Time Series", linewidth=1.5)
plt.scatter(centers, paa_values, color="red", zorder=3, label="PAA Segments")

# Annotate each PAA segment with the corresponding SAX symbol.
for center, value, symbol in zip(centers, paa_values, our_sax_word):
    va = 'bottom' if value < np.mean(ts_example) else 'top'
    plt.text(center, value, symbol, ha='center', va=va, fontsize=16, color='blue')

# Plot horizontal lines for Gaussian breakpoints (these are the thresholds used in SAX)
for b in bins:
    plt.axhline(y=b, color='green', linestyle='--', linewidth=0.5,
                label="Breakpoints" if b == bins[0] else None)

plt.xlabel("Time", fontsize=14)
plt.ylabel("Magnitude", fontsize=14)
plt.title("Coffee Time Series with PAA and SAX Annotations", fontsize=16)
plt.legend()
plt.show()
