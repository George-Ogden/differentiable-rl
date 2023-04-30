import numpy as np
import random

from typing import Any, List, Tuple

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

    def add(self, error: float, sample: Any):
        """store a sample in the buffer"""
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n: int) -> Tuple[List[Any], List[int], List[float]]:
        """sample a batch of data from experience replay"""
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx: int, error: float):
        """update priority of a sample"""
        p = self._get_priority(error)
        self.tree.update(idx, p)