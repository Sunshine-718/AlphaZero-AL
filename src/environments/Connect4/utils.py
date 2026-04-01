#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  05:58
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


def inspect(net, board=None):
    if board is None:
        board = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        state0 = board_to_state(board, 1)
        probs0, wdl0, _ = net.predict(state0)
        probs0 = probs0.flatten()
        value0 = float(wdl0[0, 1] - wdl0[0, 2])  # W - L (to-move)
        board[5, 3] = 1
        state1 = board_to_state(board, -1)
        probs1, wdl1, _ = net.predict(state1)
        probs1 = probs1.flatten()
        value1 = float(wdl1[0, 1] - wdl1[0, 2])  # W - L (to-move)
    for (idx, pX), (_, pO) in zip(enumerate(probs0), enumerate(probs1)):
        print_row(idx, pX, pO, np.max(probs0), np.max(probs1))
    print(f'State-value X: {value0: .4f}\nState-value O: {value1: .4f}')
    return probs0, value0, probs1, value1


def augment(batch):
    (state, prob, winner, steps_to_end, aux_target, root_wdl,
     valid_mask, future_root_wdl, ownership_target) = batch

    state_flipped = torch.flip(state, dims=[3])
    prob_flipped = torch.flip(prob, dims=[1])
    valid_mask_flipped = torch.flip(valid_mask, dims=[1])
    ownership_target_flipped = torch.flip(ownership_target, dims=[2])

    state = torch.cat([state, state_flipped], dim=0)
    prob = torch.cat([prob, prob_flipped], dim=0)
    winner = torch.cat([winner, winner], dim=0)
    steps_to_end = torch.cat([steps_to_end, steps_to_end], dim=0)
    aux_target = torch.cat([aux_target, aux_target], dim=0)
    root_wdl = torch.cat([root_wdl, root_wdl], dim=0)
    valid_mask = torch.cat([valid_mask, valid_mask_flipped], dim=0)
    future_root_wdl = torch.cat([future_root_wdl, future_root_wdl], dim=0)
    ownership_target = torch.cat([ownership_target, ownership_target_flipped], dim=0)
    return (state, prob, winner, steps_to_end, aux_target, root_wdl,
            valid_mask, future_root_wdl, ownership_target)


def print_row(action, probX, probO, max_X, max_O):
    print('⭐️ ' if probX == max_X else '   ', end='')
    print(f'action: {action}, prob_X: {probX * 100: 02.2f}%', end='\t')
    print('⭐️ ' if probO == max_O else '   ', end='')
    print(f'action: {action}, prob_O: {probO * 100: 02.2f}%')
