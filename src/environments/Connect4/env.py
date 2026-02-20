#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024 13:03
# Modified for Incremental Check
import numpy as np
from ..Environment import Environment


class Env(Environment):
    NUM_SYMMETRIES = 2  # 0=identity, 1=horizontal flip

    def __init__(self, board=None):
        super().__init__()
        self.board = np.zeros((6, 7), dtype=np.float32) if board is None else board
        # 记录最后一步棋 (row, col)，初始为 None
        self.last_move = None

        # 只有在传入 board 时才需要计算 turn
        if board is not None:
            count = np.count_nonzero(self.board)
            self.turn = 1 if count % 2 == 0 else -1
        else:
            self.turn = 1

    def copy(self):
        new_env = Env()
        new_env.board = np.copy(self.board)
        new_env.turn = self.turn
        new_env.last_move = self.last_move  # 复制最后落子状态
        return new_env

    def reset(self):
        self.board.fill(0)
        self.turn = 1
        self.last_move = None
        return self.board

    def done(self):
        return (self.winPlayer() != 0) or self.check_full()

    def valid_move(self):
        # 优化：只检查第一行
        return [i for i in range(7) if self.board[0, i] == 0]

    def valid_mask(self):
        # 优化：只检查第一行
        return (self.board[0, :] == 0).tolist()

    def switch_turn(self):
        self.turn = -1 if self.turn == 1 else 1
        return self.turn

    def place(self, action):
        """落子并记录位置"""
        # 从底向上找
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.turn
                self.last_move = (row, action)  # [关键] 更新增量检查点
                return True
        return False

    def check_full(self):
        # 优化：检查顶行是否有空
        return np.all(self.board[0] != 0)

    def winPlayer(self):
        """增量检查获胜者"""
        if self.last_move is None:
            return 0

        r, c = self.last_move
        player = self.board[r, c]
        if player == 0:
            return 0

        rows, cols = 6, 7

        # 方向：横、竖、对角1、对角2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # 正向延伸
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols and self.board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            # 反向延伸
            nr, nc = r - dr, c - dc
            while 0 <= nr < rows and 0 <= nc < cols and self.board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 4:
                return int(player)

        return 0

    def current_state(self):
        # 生成 (1, 3, 6, 7) 格式
        state = np.zeros((1, 3, 6, 7), dtype=np.float32)
        state[0, 0] = (self.board == 1)
        state[0, 1] = (self.board == -1)
        state[0, 2] = self.turn
        return state

    def step(self, action):
        if self.place(action):
            self.switch_turn()

    def show(self):
        board_int = self.board.astype(int)
        print('=' * 20)
        for row in board_int:
            line = []
            for val in row:
                if val == 0:
                    line.append('_')
                elif val == 1:
                    line.append('X')
                else:
                    line.append('O')
            print(' '.join(line))
        print('0 1 2 3 4 5 6')
        print('=' * 20)

    def apply_symmetry(self, sym_id: int, inplace: bool = False):
        """sym_id=0: identity, sym_id=1: horizontal flip"""
        target = self if inplace else self.copy()
        if sym_id == 0:
            return target
        target.board = target.board[:, ::-1]
        if target.last_move is not None:
            r, c = target.last_move
            target.last_move = (r, 6 - c)
        return target

    @staticmethod
    def inverse_symmetry_action(sym_id: int, col: int) -> int:
        """对 action 应用对称逆变换"""
        return col if sym_id == 0 else 6 - col

    def random_symmetry(self):
        """随机选一种对称变换，返回 (env_copy, sym_id)"""
        sym_id = np.random.randint(0, self.NUM_SYMMETRIES)
        return self.apply_symmetry(sym_id), sym_id
