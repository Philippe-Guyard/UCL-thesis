import numpy as np

class SimpleHistogram:
    def __init__(self, precision=0.001, min_range=-1, max_range=1):
        self.precision = precision
        self.min_range = min_range
        self.max_range = max_range
        self.bin_count = int((max_range - min_range) / precision)
        self.histogram = np.zeros(self.bin_count)
        self._write_mode = True
        self.cumsum = None

    def update(self, value):
        assert self._write_mode
        """Update histogram count based on the value."""
        if self.min_range <= value < self.max_range:
            bin_index = int((value - self.min_range) / self.precision)
            self.histogram[bin_index] += 1
    
    def for_reading(self):
        self._write_mode = False
        self.cumsum = self.histogram.cumsum()
        return self

    def get_histogram(self):
        return self.histogram 

    def inv_quantile(self, value):
        """Compute the quantile of a given number based on the histogram."""
        if value < self.min_range:
            return 0.0
        elif value >= self.max_range:
            return 1.0

        bin_index = int((value - self.min_range) / self.precision)
        
        # Compute CDF up to the bin containing the value
        total_counts = self.cumsum[-1]
        cdf_value = self.cumsum[bin_index - 1] / total_counts if bin_index > 0 else 0.0
        return cdf_value

    def quantile(self, quantile):
        """Compute the quantile value from the histogram."""
        if not (0 <= quantile <= 1):
            raise ValueError("Quantile should be between 0 and 1.")

        cumulative_counts = self.cumsum 
        total_counts = cumulative_counts[-1]
        target_count = quantile * total_counts

        # Find the first bin where the cumulative count exceeds the target count
        bin_index = np.searchsorted(cumulative_counts, target_count)
        quantile_value = self.min_range + bin_index * self.precision
        return quantile_value

    def dump_to_file(self, filename):
        """Save the histogram data to a file."""
        np.save(filename, self.histogram)

    def load_from_file(self, filename):
        """Load the histogram data from a file."""
        self.histogram = np.load(filename)
        # Ensure the loaded histogram has the correct size
        assert len(self.histogram) == self.bin_count, "Loaded histogram size mismatch."
