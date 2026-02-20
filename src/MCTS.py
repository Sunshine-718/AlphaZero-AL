#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jun/2025  21:55
import math
import numpy as np
from .Cache import LRUCache as Cache


class TreeNode:
    def __init__(self, parent, prior, discount, dirichlet_noise=None, eps=0.25):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.prior = prior
        self.discount = discount
        self.noise = dirichlet_noise if dirichlet_noise is not None else prior
        self.eps = eps
        self.deterministic = False

    def train(self):
        if self.deterministic is True:
            if not self.children:
                self.deterministic = False
                return
            for node in self.children.values():
                node.train()
                self.deterministic = False

    def eval(self):
        if self.deterministic is False:
            if not self.children:
                self.deterministic = True
                return
            for node in self.children.values():
                node.eval()
                self.deterministic = True

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_root(self):
        return self.parent is None

    def PUCT(self, c_init, c_base, fpu_value=0.0):
        if self.parent.is_root:
            prior = (1 - self.eps) * self.prior + self.eps * self.noise
        else:
            prior = self.prior
        self.u = (c_init + math.log((1 + self.parent.n_visits + c_base) / c_base)
                  ) * prior * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        q_value = fpu_value if self.n_visits == 0 else -self.Q
        return q_value + self.u

    def UCT(self, c_init, c_base):
        if self.n_visits == 0:
            self.u = float('inf')
        else:
            self.u = (c_init + math.log((1 + self.parent.n_visits + c_base) / c_base)
                      ) * math.sqrt(math.log(self.parent.n_visits) / self.n_visits)
        return -self.Q + self.u

    def UCB(self, c_init, c_base, UCT=False, fpu_value=0.0):
        if UCT:
            return self.UCT(c_init, c_base)
        else:
            return self.PUCT(c_init, c_base, fpu_value)

    def expand(self, action_probs, noise=None):
        for idx, (action, prior) in enumerate(action_probs):
            if action not in self.children:
                if self.deterministic or noise is None:
                    self.children[action] = TreeNode(self, prior, self.discount, None, self.eps)
                else:
                    self.children[action] = TreeNode(self, prior, self.discount, noise[idx], self.eps)

    def select(self, c_init, c_base, UCT=False, fpu_reduction=0.4):
        if UCT:
            fpu_value = 0.0
        else:
            seen_policy = sum(c.prior for c in self.children.values() if c.n_visits > 0)
            fpu_value = self.Q - fpu_reduction * math.sqrt(seen_policy)
        return max(self.children.items(), key=lambda action_node: action_node[1].UCB(c_init, c_base, UCT, fpu_value))

    def update(self, leaf_value):
        if self.parent:
            self.parent.update(-leaf_value * self.discount)
        self.n_visits += 1
        # Q = ((n - 1) * Q_old + leaf_value) / n
        self.Q += (leaf_value - self.Q) / self.n_visits


class MCTS:
    def __init__(self, policy_value_fn, c_init, n_playout, discount, alpha, eps=0.25, fpu_reduction=0.4, use_symmetry=True):
        self.root = TreeNode(None, 1, discount, None, eps)
        self.policy = policy_value_fn
        self.c_init = c_init
        self.c_base = 500
        self.n_playout = n_playout
        self.alpha = alpha
        self.eps = eps
        self.deterministic = False
        self.fpu_reduction = fpu_reduction
        self.use_symmetry = use_symmetry

    def train(self):
        self.root.train()
        self.deterministic = False

    def eval(self):
        self.root.eval()
        self.deterministic = True

    def select_leaf_node(self, env, UCT=False):
        node = self.root
        while not node.is_leaf:
            action, node = node.select(self.c_init, self.c_base, UCT, self.fpu_reduction)
            env.step(action)
        return node

    def playout(self, env):
        node = self.select_leaf_node(env, True)
        if self.use_symmetry:
            env_aug, sym_id = env.random_symmetry()
        else:
            env_aug, sym_id = env.copy(), 0
        action_probs, leaf_value = self.policy(env_aug)
        if sym_id != 0:
            action_probs = [(env.inverse_symmetry_action(sym_id, action), prob) for action, prob in action_probs]
        if not env.done():
            node.expand(action_probs)
        node.update(leaf_value)

    def get_action(self, env):
        for _ in range(self.n_playout):
            self.playout(env.copy())
        return max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits)

    def prune_root(self, node_index):
        if node_index in self.root.children:
            self.root = self.root.children[node_index]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1, self.root.discount)


class MCTS_AZ(MCTS):
    def __init__(self, policy_value_fn, c_init, n_playout, discount, alpha, cache_size, eps=0.25, fpu_reduction=0.4, use_symmetry=True):
        super().__init__(policy_value_fn, c_init, n_playout, discount, alpha, eps, fpu_reduction, use_symmetry)
        self.cache = Cache(cache_size)
        self.use_cache = cache_size > 0

    def refresh_cache(self):
        if self.use_cache:
            self.cache.refresh(self.policy.predict)

    def playout(self, env):
        noise = None
        node = self.select_leaf_node(env)
        if self.use_symmetry:
            env_aug, sym_id = env.random_symmetry()
        else:
            env_aug, sym_id = env.copy(), 0
        #
        valid = env_aug.valid_move()
        state = env_aug.current_state()
        if self.use_cache and (state in self.cache):
            probs, value = self.cache.get(state)
        else:
            probs, value = self.policy.predict(state)
            if self.use_cache:
                self.cache.put(state, (probs, value))
        probs = probs.flatten()[valid]
        probs /= sum(probs)
        action_probs = tuple(zip(valid, probs))
        leaf_value = value.flatten()[0]
        #
        if sym_id != 0:
            action_probs = [(env.inverse_symmetry_action(sym_id, action), prob) for action, prob in action_probs]
        if not env.done():
            if self.alpha is not None and not self.deterministic:
                noise = np.random.dirichlet([self.alpha for _ in action_probs])
            node.expand(action_probs, noise)
        else:
            winner = env.winPlayer()
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = (1 if winner == env.turn else -1)
        node.update(leaf_value)

    def get_action_visits(self, env):
        assert ((self.alpha is not None) or (self.alpha is None and self.deterministic))
        for _ in range(self.n_playout):
            self.playout(env.copy())
        act_visits = [(action, node.n_visits) for action, node in self.root.children.items()]
        return zip(*act_visits)
