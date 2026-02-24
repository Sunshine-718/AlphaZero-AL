from collections import OrderedDict
import numpy as np


class LRUCache:
    def __init__(self, capacity=None):
        match capacity:
            case None:
                self._cap = float('inf')
            case n if int(n) == n and n >= 0:
                self._cap = n
            case _:
                raise ValueError
        self._od = OrderedDict()

    @staticmethod
    def hash_ndarray(ndarray):
        if isinstance(ndarray, np.ndarray):
            array = np.ascontiguousarray(ndarray)
            return array.tobytes()
        elif isinstance(ndarray, bytes):
            return ndarray
        else:
            raise ValueError

    def refresh(self, pv_func):
        if len(self) == 0:
            return
        keys = [key for key in self._od.keys()]
        states = [self._od[key]['state'] for key in keys]
        states = np.concatenate(states, axis=0)
        probs, values, moves_left = pv_func(states)
        for idx, key in enumerate(keys):
            value = (probs[idx].reshape(1, -1), values[idx].reshape(1, -1), moves_left[idx].reshape(1, -1))
            self.put(key, value)

    def __contains__(self, key):
        key = self.hash_ndarray(key)
        return key in self._od.keys()

    def get(self, key):
        key = self.hash_ndarray(key)
        self._od.move_to_end(key, last=False)
        return self._od[key]['value']

    def put(self, key, value):
        hashed_key = self.hash_ndarray(key)
        if hashed_key in self._od:
            self._od[hashed_key]['value'] = value
            self._od.move_to_end(hashed_key, last=False)
        else:
            self._od[hashed_key] = {'state': key, 'value': value}
            self._od.move_to_end(hashed_key, last=False)
            if len(self._od) > self._cap:
                self._od.popitem(last=True)

    def __len__(self):
        return len(self._od)


class LFUCache:
    def __init__(self, capacity=None):
        match capacity:
            case None:
                self._cap = float('inf')
            case n if int(n) == n and n > 0:
                self._cap = int(n)
            case _:
                raise ValueError

        self._data: dict[bytes, dict] = {}
        self._freq: dict[bytes, int] = {}
        self._buckets: dict[int, OrderedDict] = {}
        self._min_freq: int = 0

    @staticmethod
    def hash_ndarray(ndarray):
        if isinstance(ndarray, np.ndarray):
            array = np.ascontiguousarray(ndarray)
            return array.tobytes()
        elif isinstance(ndarray, (bytes, bytearray, memoryview)):
            return bytes(ndarray)
        else:
            raise ValueError

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        h = self.hash_ndarray(key)
        return h in self._data

    def _touch(self, h: bytes):
        f = self._freq[h]
        bucket = self._buckets[f]
        if h in bucket:
            del bucket[h]
        if not bucket:
            del self._buckets[f]
            if self._min_freq == f:
                self._min_freq = f + 1

        nf = f + 1
        self._freq[h] = nf
        if nf not in self._buckets:
            self._buckets[nf] = OrderedDict()
        self._buckets[nf][h] = None

    def _evict_if_needed(self):
        if self._cap == float('inf') or len(self._data) < self._cap:
            return
        f = self._min_freq
        bucket = self._buckets.get(f)
        if not bucket:
            f = min(self._buckets.keys())
            bucket = self._buckets[f]
        h, _ = bucket.popitem(last=False)
        if not bucket:
            del self._buckets[f]
        del self._data[h]
        del self._freq[h]

    def get(self, key):
        h = self.hash_ndarray(key)
        self._touch(h)
        return self._data[h]['value']

    def put(self, key, value):
        h = self.hash_ndarray(key)
        if h in self._data:
            self._data[h]['value'] = value
            self._touch(h)
            return

        self._evict_if_needed()

        self._data[h] = {'state': key, 'value': value}
        self._freq[h] = 1
        if 1 not in self._buckets:
            self._buckets[1] = OrderedDict()
        self._buckets[1][h] = None
        self._min_freq = 1

    def refresh(self, pv_func):
        if len(self) == 0:
            return
        keys = list(self._data.keys())
        states = [self._data[h]['state'] for h in keys]
        states = np.concatenate(states, axis=0)
        probs, values, moves_left = pv_func(states)
        for idx, h in enumerate(keys):
            self._data[h]['value'] = (probs[idx].reshape(1, -1),
                                      values[idx].reshape(1, -1),
                                      moves_left[idx].reshape(1, -1))
