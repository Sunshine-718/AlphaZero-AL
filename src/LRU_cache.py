from collections import OrderedDict
import numpy as np


class LRUCache:
    def __init__(self, capacity=None):
        match capacity:
            case None:
                self._cap = float('inf')
            case n if int(n) == n and n > 0:
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
        probs, values = pv_func(states)
        for idx, key in enumerate(keys):
            value = (probs[idx].reshape(1, -1), values[idx].reshape(1, -1))
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
