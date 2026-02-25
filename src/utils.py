#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 21/Jul/2024  03:52
import numpy as np


class Elo:
    def __init__(self, init_A=1500, init_B=1500):
        self.R_A = init_A
        self.R_B = init_B

    def update(self, result_a, k=32):
        """
        :param result_a: 1 (A win), 0.5 (draw), 0 (A lose).
        :param k: factor, default: 32.
        """
        expected_a = 1 / (1 + pow(10, (self.R_B - self.R_A) / 400))
        expected_b = 1 / (1 + pow(10, (self.R_A - self.R_B) / 400))
        self.R_A = max(self.R_A + k * (result_a - expected_a), 1500)
        self.R_B = max(self.R_B + k * ((1 - result_a) - expected_b), 1500)
        return self.R_A, self.R_B


def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)
