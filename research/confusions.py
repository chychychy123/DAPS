"""Disk-backed per-image confusion matrices for complete VOC/COCO evaluation."""
from pathlib import Path
import numpy as np


class ConfusionStore:
    def __init__(self, output, count, classes):
        self.output = Path(output)
        self.scratch = self.output.with_suffix('.working.npy')
        self.values = np.lib.format.open_memmap(self.scratch, mode='w+', dtype=np.int32,
                                               shape=(count, classes, classes))
        self.count = 0

    def append(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.shape != self.values.shape[1:] or (matrix < 0).any() or (matrix > np.iinfo(np.int32).max).any():
            raise ValueError('Invalid per-image confusion counts')
        self.values[self.count] = matrix
        self.count += 1

    def export(self, ids):
        if self.count != len(ids) or self.count != len(self.values):
            raise ValueError('Incomplete evaluation coverage')
        self.values.flush()
        # Passing the memmap directly avoids constructing a multi-GB RAM array.
        np.savez_compressed(self.output, ids=np.asarray(ids), matrices=self.values)
        self.values._mmap.close()
        self.values = None
        self.scratch.unlink()
