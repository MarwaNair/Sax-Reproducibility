import numpy as np
import scipy.stats
from pyts.approximation import PiecewiseAggregateApproximation

class SAX:
    def __init__(self, word_size=8, alphabet_size=3):

        self.word_size = word_size # Number of PAA segments (w)
        self.alphabet_size = alphabet_size # Number of discrete symbols (a)
        self.breakpoints = self._compute_breakpoints()
        self.distance_table = self._compute_distance_table()
        self.time_series_length = None  # Updated during SAX transform

    def _compute_breakpoints(self):
        """Compute breakpoints for the given alphabet size using a standard normal distribution."""
        
        return scipy.stats.norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
    
    def _compute_distance_table(self):
        """Precompute symbol-to-symbol distances based on breakpoints."""
        dist_table = np.zeros((self.alphabet_size, self.alphabet_size))
        for i in range(self.alphabet_size):
            for j in range(i + 1, self.alphabet_size):
                dist_table[i, j] = self.breakpoints[j - 1] - self.breakpoints[i]
                dist_table[j, i] = dist_table[i, j]  # Symmetric
        return dist_table

    def z_normalize(self, series):
        """Normalize the time series to have zero mean and unit variance."""
        mean, std = np.mean(series), np.std(series)
        return (series - mean) / std if std > 0 else np.zeros_like(series)

    
    def paa_transform(self, series):
        """Convert the time series into a PAA representation."""
        paa = PiecewiseAggregateApproximation(window_size=None, output_size=self.word_size)
        return paa.transform(series.reshape(1, -1))[0]

    def sax_transform(self, series):
        """Convert the time series into a SAX representation."""
        self.time_series_length = len(series)  # Update time series length
        normalized_series = self.z_normalize(series)
        paa_values = self.paa_transform(normalized_series)
        sax_word = self._map_to_symbol(paa_values)
        return "".join(sax_word)  

    def _map_to_symbol(self, paa_values):
        """Assign a symbol to each PAA value."""
        indices = np.searchsorted(self.breakpoints, paa_values, side='left')
        return [chr(97 + i) for i in indices]
    
    def mindist(self, sax_word1, sax_word2):
        """Compute MINDIST between two SAX words."""
        if len(sax_word1) != len(sax_word2):
            raise ValueError("SAX words must have the same length")

        # Convert characters to indices 
        indices1 = [ord(ch) - 97 for ch in sax_word1]
        indices2 = [ord(ch) - 97 for ch in sax_word2]

        squared_distances = np.array([
            self.distance_table[indices1[i], indices2[i]] ** 2
            for i in range(self.word_size)
        ])
        scaling_factor = self.time_series_length / self.word_size
        mindist_value = np.sqrt(scaling_factor * np.sum(squared_distances))
        return mindist_value

if __name__ == "__main__":
    
    # Sample time series to test 
    ts = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38])
    
    sax = SAX(word_size=4, alphabet_size=3)
    
    # Compute SAX representation
    sax_word = sax.sax_transform(ts)
    print("SAX Representation:", sax_word)
