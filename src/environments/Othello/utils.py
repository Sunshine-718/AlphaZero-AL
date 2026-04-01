import torch
import numpy as np
from numba import njit


@njit(fastmath=True)
def board_to_state(board, turn):
    temp = np.zeros((1, 3, board.shape[0], board.shape[1]), dtype=np.float32)
    # Relative perspective:
    #   ch0 = side-to-move stones, ch1 = opponent stones
    temp[:, 0] = board == turn
    temp[:, 1] = board == -turn
    temp[:, 2] = np.ones((board.shape[0], board.shape[1]), dtype=np.float32) * turn
    return temp


@njit(fastmath=True)
def check_full(board):
    return len(np.where(board == 0)[0]) == 0


_SYM_TRANSFORMS = [
    lambda r, c: (r, c),
    lambda r, c: (c, 7 - r),
    lambda r, c: (7 - r, 7 - c),
    lambda r, c: (7 - c, r),
    lambda r, c: (r, 7 - c),
    lambda r, c: (7 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (7 - c, 7 - r),
]


def _build_perm(sym_id):
    perm = torch.zeros(64, dtype=torch.long)
    transform = _SYM_TRANSFORMS[sym_id]
    for i in range(64):
        nr, nc = transform(i // 8, i % 8)
        perm[i] = nr * 8 + nc
    return perm


def _apply_sym_state(state, sym_id):
    if sym_id == 0:
        return state
    perm = _build_perm(sym_id)
    n, c, h, w = state.shape
    flat = state.view(n, c, 64)
    result = torch.zeros_like(flat)
    result[:, :, perm] = flat
    return result.view(n, c, h, w)


def _apply_sym_policy(policy, sym_id):
    if sym_id == 0:
        return policy
    perm = _build_perm(sym_id)
    board_part = policy[:, :64]
    pass_part = policy[:, 64:]
    result = torch.zeros_like(board_part)
    result[:, perm] = board_part
    return torch.cat([result, pass_part], dim=1)


def _apply_sym_board_target(board_target, sym_id):
    if sym_id == 0:
        return board_target
    perm = _build_perm(sym_id)
    flat = board_target.view(board_target.shape[0], 64)
    result = torch.zeros_like(flat)
    result[:, perm] = flat
    return result.view_as(board_target)


def augment(batch):
    """Apply the four legal Othello symmetries that preserve the initial setup."""
    (state, prob, winner, steps_to_end, aux_target, root_wdl,
     valid_mask, future_root_wdl, ownership_target) = batch

    states_all = [state]
    probs_all = [prob]
    masks_all = [valid_mask]
    future_all = [future_root_wdl]
    ownership_all = [ownership_target]

    for sym_id in (2, 6, 7):
        states_all.append(_apply_sym_state(state, sym_id))
        probs_all.append(_apply_sym_policy(prob, sym_id))
        masks_all.append(_apply_sym_policy(valid_mask.float(), sym_id).bool())
        future_all.append(future_root_wdl)
        ownership_all.append(_apply_sym_board_target(ownership_target, sym_id))

    state = torch.cat(states_all, dim=0)
    prob = torch.cat(probs_all, dim=0)
    winner = winner.repeat(4, 1)
    steps_to_end = steps_to_end.repeat(4, 1)
    aux_target = aux_target.repeat(4, 1)
    root_wdl = root_wdl.repeat(4, 1)

    valid_mask = torch.cat(masks_all, dim=0)
    future_root_wdl = torch.cat(future_all, dim=0)
    ownership_target = torch.cat(ownership_all, dim=0)
    return (state, prob, winner, steps_to_end, aux_target, root_wdl,
            valid_mask, future_root_wdl, ownership_target)


def inspect(net, board=None):
    if board is None:
        board = np.zeros((8, 8), dtype=np.float32)
        board[3, 3] = -1
        board[3, 4] = 1
        board[4, 3] = 1
        board[4, 4] = -1
    with torch.no_grad():
        from src.environments.Othello import Env

        state0 = board_to_state(board, 1)
        env0 = Env(board.astype(np.float32))
        mask0 = np.asarray(env0.valid_mask(), dtype=bool)[None, :]
        probs0, wdl0, _ = net.predict(state0, action_mask=mask0)
        probs0 = probs0.flatten()
        value0 = float(wdl0[0, 1] - wdl0[0, 2])

        board[2, 3] = 1
        board[3, 3] = 1
        state1 = board_to_state(board, -1)
        env1 = Env(board.astype(np.float32))
        mask1 = np.asarray(env1.valid_mask(), dtype=bool)[None, :]
        probs1, wdl1, _ = net.predict(state1, action_mask=mask1)
        probs1 = probs1.flatten()
        value1 = float(wdl1[0, 1] - wdl1[0, 2])

    top_x = np.argsort(probs0)[::-1][:10]
    top_o = np.argsort(probs1)[::-1][:10]
    print("Black (P1) top actions:")
    for idx in top_x:
        label = f"({idx // 8},{idx % 8})" if idx < 64 else "pass"
        print(f"  {label}: {probs0[idx] * 100:.2f}%")
    print(f"State-value Black: {value0:.4f}\n")
    print("White (P2) top actions:")
    for idx in top_o:
        label = f"({idx // 8},{idx % 8})" if idx < 64 else "pass"
        print(f"  {label}: {probs1[idx] * 100:.2f}%")
    print(f"State-value White: {value1:.4f}")
    return probs0, value0, probs1, value1
