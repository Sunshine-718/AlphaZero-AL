#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symmetry ensemble utilities for MCTS search.

Provides board/action permutation tables and transform/inverse functions
used by AlphaZeroPlayer's sym_ensemble mode. Each game registers its
symmetry group (a subset of the full symmetry group that preserves the
initial position) along with precomputed permutation arrays.

Supported games:
  - Othello: Klein four-group {identity, 180° rotation, main diagonal, anti-diagonal}
  - Connect4: {identity, horizontal flip (column mirror)}
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Othello — Klein four-group (4 symmetries)
# ═══════════════════════════════════════════════════════════════════════

_OTHELLO_SYM_IDS = [0, 2, 6, 7]

_OTHELLO_TRANSFORMS = [
    lambda r, c: (r, c),         # 0: identity
    lambda r, c: (c, 7 - r),     # 1: CW 90°
    lambda r, c: (7 - r, 7 - c), # 2: 180°
    lambda r, c: (7 - c, r),     # 3: CW 270°
    lambda r, c: (r, 7 - c),     # 4: horizontal flip
    lambda r, c: (7 - r, c),     # 5: vertical flip
    lambda r, c: (c, r),         # 6: main diagonal
    lambda r, c: (7 - c, 7 - r), # 7: anti-diagonal
]

_OTHELLO_PERMS = {}
for _sid in _OTHELLO_SYM_IDS:
    _perm = np.empty(64, dtype=np.intp)
    for _i in range(64):
        _nr, _nc = _OTHELLO_TRANSFORMS[_sid](_i // 8, _i % 8)
        _perm[_i] = _nr * 8 + _nc
    _OTHELLO_PERMS[_sid] = _perm

# ═══════════════════════════════════════════════════════════════════════
#  Connect4 — {identity, horizontal flip}
# ═══════════════════════════════════════════════════════════════════════

_CONNECT4_SYM_IDS = [0, 1]

_CONNECT4_BOARD_PERM = np.empty(42, dtype=np.intp)  # 6×7
for _r in range(6):
    for _c in range(7):
        _CONNECT4_BOARD_PERM[_r * 7 + _c] = _r * 7 + (6 - _c)
_CONNECT4_ACTION_PERM = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.intp)

# ═══════════════════════════════════════════════════════════════════════
#  Unified per-game symmetry registry
# ═══════════════════════════════════════════════════════════════════════

SYM_REGISTRY = {
    'Othello': {
        'sym_ids': _OTHELLO_SYM_IDS,
        'board_shape': (8, 8),
        'board_perms': _OTHELLO_PERMS,
        'action_size': 65,
        'board_cells': 64,
    },
    'Connect4': {
        'sym_ids': _CONNECT4_SYM_IDS,
        'board_shape': (6, 7),
        'board_perms': {1: _CONNECT4_BOARD_PERM},
        'action_size': 7,
        'board_cells': 7,
        'action_perm': {1: _CONNECT4_ACTION_PERM},
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def get_sym_config(game_name):
    """Return symmetry config dict for *game_name*, or None."""
    return SYM_REGISTRY.get(game_name)


def apply_sym_board(board, sym_id, game_name):
    """Apply symmetry *sym_id* to a 2-D board array (returns a copy)."""
    if sym_id == 0:
        return board.copy()
    reg = SYM_REGISTRY[game_name]
    perm = reg['board_perms'][sym_id]
    H, W = reg['board_shape']
    result = np.empty(H * W, dtype=board.dtype)
    result[perm] = board.ravel()
    return result.reshape(H, W)


def apply_sym_action(action, sym_id, game_name):
    """Map an action index through symmetry *sym_id*.

    All symmetries used here are self-inverse (involutions), so forward
    and inverse transforms are identical.  For Othello, the pass action
    (index 64) is invariant under all symmetries.
    """
    if sym_id == 0:
        return action
    reg = SYM_REGISTRY[game_name]
    # Connect4: action = column index, use dedicated action_perm
    if 'action_perm' in reg:
        return int(reg['action_perm'][sym_id][action])
    # Othello: action 0..63 = board cell, 64 = pass
    board_cells = reg['board_cells']
    if action >= board_cells:
        return action  # pass is invariant
    return int(reg['board_perms'][sym_id][action])


def inverse_sym_visits(visits, sym_id, game_name):
    """Inverse-permute a 1-D action-visit vector (self-inverse symmetries)."""
    if sym_id == 0:
        return visits.copy()
    reg = SYM_REGISTRY[game_name]
    # Connect4: actions are column indices — use dedicated action_perm
    if 'action_perm' in reg:
        perm = reg['action_perm'][sym_id]
        result = np.empty_like(visits)
        result[perm] = visits
        return result
    # Othello: action = board_cell index (0..63) + pass (64)
    board_cells = reg['board_cells']
    perm = reg['board_perms'][sym_id]
    board_part = visits[:board_cells]
    result = np.empty(board_cells, dtype=visits.dtype)
    result[perm] = board_part
    if len(visits) > board_cells:
        return np.append(result, visits[board_cells:])
    return result


def inverse_sym_stats(raw_stats, sym_ids, game_name):
    """Inverse-transform and aggregate per-action root stats across K symmetric trees.

    Args:
        raw_stats: dict from ``get_root_stats()``, per-action arrays shape ``(K, A)``
        sym_ids: list of sym_id for each of the K trees
        game_name: game name for symmetry lookup

    Returns:
        dict with same keys, per-action values shape ``(1, A)`` — aggregated to
        the canonical (identity) perspective.
    """
    per_action_keys = ['N', 'Q', 'prior', 'noise', 'M', 'D', 'P1W', 'P2W']
    root_keys = ['root_N', 'root_Q', 'root_M', 'root_D', 'root_P1W', 'root_P2W']
    result = {}

    # Root-level stats: average across trees
    for k in root_keys:
        result[k] = raw_stats[k].mean(axis=0, keepdims=True)

    K = len(sym_ids)
    A = raw_stats['N'].shape[1]

    for k in per_action_keys:
        data = raw_stats[k]  # (K, A)
        transformed = np.zeros((K, A), dtype=data.dtype)
        for i, sid in enumerate(sym_ids):
            transformed[i] = inverse_sym_visits(data[i], sid, game_name)
        if k == 'N':
            result[k] = transformed.sum(axis=0, keepdims=True)
        elif k in ('Q', 'M', 'D', 'P1W', 'P2W'):
            # Visit-weighted average
            weights = np.zeros((K, A), dtype=np.float32)
            for i, sid in enumerate(sym_ids):
                weights[i] = inverse_sym_visits(
                    raw_stats['N'][i].astype(np.float32), sid, game_name
                )
            total_w = weights.sum(axis=0)
            mask = total_w > 0
            avg = np.zeros(A, dtype=np.float32)
            avg[mask] = (transformed * weights).sum(axis=0)[mask] / total_w[mask]
            result[k] = avg.reshape(1, A)
        else:
            # prior, noise: simple average
            result[k] = transformed.mean(axis=0, keepdims=True)

    return result
