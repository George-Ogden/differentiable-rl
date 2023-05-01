import numpy as np

from src.buffer import Buffer

def test_buffer_ids():
    buffer = Buffer(11)
    for i in range(1, 11):
        buffer.add(i, i)
    assert len(buffer) == 10
    
    importances = {i: i for i in range(1, 11)}
    mapping = {}
    for _ in range(10):
        values, ids, _ = buffer.sample(5)
        for id, value in zip(ids, values):
            mapping[id] = value
            buffer.update(id, importances[value] / 2)
            importances[value] /= 2

    assert len(mapping) == 10
    assert len(set(mapping.keys())) == 10
    assert len(set(mapping.values())) == 10


def test_buffer_importances():
    buffer = Buffer(6)
    for i in range(1, 6):
        buffer.add(i, i)
    assert len(buffer) == 5
    
    importances = {i: i for i in range(1, 6)}
    mapping = {}
    for _ in range(10):
        values, ids, weights = buffer.sample(5)
        ordering = np.argsort(weights)
        values = np.array(values)[ordering]
        ids = np.array(ids)[ordering]
        weights = weights[ordering]
        for x, y in zip(values[:-1], values[1:]):
            assert importances[x] >= importances[y]

        for id, value in zip(ids, values):
            mapping[id] = value
            buffer.update(id, importances[value] / 2)
            importances[value] /= 2

def test_full_buffer():
    buffer = Buffer(7)
    buffer.add(np.arange(1, 8), np.arange(1, 8, dtype=int))
    assert len(buffer) == 7
    
    importances = {i: i for i in range(1, 8)}
    mapping = {}
    for _ in range(10):
        values, ids, weights = buffer.sample(5)
        ordering = np.argsort(weights)
        values = np.array(values)[ordering]
        ids = np.array(ids)[ordering]
        weights = weights[ordering]
        for x, y in zip(values[:-1], values[1:]):
            assert importances[x] >= importances[y]

        for id, value in zip(ids, values):
            mapping[id] = value
            buffer.update(id, importances[value] / 2)
            importances[value] /= 2
    
    assert len(mapping) == 7
    assert len(set(mapping.keys())) == 7
    assert len(set(mapping.values())) == 7

def test_cyclic_buffer():
    buffer = Buffer(7)
    buffer.add(np.arange(1, 7), np.arange(1, 7, dtype=int))
    buffer.add(np.arange(7, 11), np.arange(7, 11, dtype=int))
    assert len(buffer) == 7
    
    importances = {i: i for i in range(4, 11)}
    mapping = {}
    for _ in range(10):
        values, ids, weights = buffer.sample(5)
        ordering = np.argsort(weights)
        values = np.array(values)[ordering]
        ids = np.array(ids)[ordering]
        weights = weights[ordering]
        for x, y in zip(values[:-1], values[1:]):
            assert importances[x] >= importances[y]

        for id, value in zip(ids, values):
            mapping[id] = value
            buffer.update(id, importances[value] / 2)
            importances[value] /= 2
    
    assert len(mapping) == 7
    assert len(set(mapping.keys())) == 7
    assert len(set(mapping.values())) == 7
    assert sorted(list(mapping.values())) == list(range(4, 11))