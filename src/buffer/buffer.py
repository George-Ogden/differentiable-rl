import numpy as np
import random

from typing import Any, List, Tuple, Union

from .sumtree import SumTree

class Buffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity: int):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error: Union[float, List[float]], sample: Union[Any, List[Any]]):
        """store a sample in the buffer"""
        error = np.array(error)
        if error.ndim == 0:
            error = np.array([error])
            sample = np.array([sample])
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n: int) -> Tuple[List[Any], List[int], List[float]]:
        """sample a batch of data from experience replay"""
        segment = self.tree.total() / n

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        a = segment * np.arange(n)
        b = segment * (np.arange(n) + 1)
        
        s = np.random.uniform(a, b)
        idxs, priorities, batch = self.tree.get(s)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx: Union[int, List[int]], error: Union[float, List[float]]):
        """update priority of a sample"""
        error = np.array(error).reshape(-1)
        idx = np.array(idx).reshape(-1)

        p = self._get_priority(error)
        self.tree.update(idx, p)