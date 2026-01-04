#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024 13:03
# Modified for Incremental Check
import numpy as np
from ..Environment import Environment


class Env(Environment):
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

    def flip(self, inplace: bool = False):
        target = self if inplace else self.copy()
        target.board = target.board[:, ::-1]

        # [关键] 翻转 last_move 的列坐标
        if target.last_move is not None:
            r, c = target.last_move
            target.last_move = (r, 6 - c)

        return target

    def flip_action(self, col: int) -> int:
        # 保持原有逻辑：如果棋盘对称则不翻转action，否则翻转
        # 这里为了简化和Cython保持一致，直接用Python实现简单检测
        # 注意：如果您的训练代码强烈依赖原有的对称检测逻辑，请保留此段
        # 如果追求速度，通常直接返回 6 - col 即可，但这取决于策略网络的训练方式
        return 6 - col

    def random_flip(self, p: float = 0.5):
        env_copy = self.copy()
        flipped = False
        if np.random.rand() < p:
            env_copy.flip(inplace=True)
            flipped = True
        return env_copy, flipped
