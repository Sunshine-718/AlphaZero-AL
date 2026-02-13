#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:55
import numpy as np


class Game:
    def __init__(self, env):
        self.env = env

    def play(self, player1, player2, show=1):
        self.env.reset()
        players = [None, player1, player2]
        if show:
            self.env.show()
        while True:
            current_turn = self.env.turn
            player = players[current_turn]
            action, *_ = player.get_action(self.env)
            player.reset_player()
            self.env.step(action)
            if show:
                self.env.show()
            if self.env.done():
                winner = self.env.winPlayer()
                if show:
                    if winner != 0:
                        print('Game end. Winner is', (None, 'X', 'O')[int(winner)])
                    else:
                        print('Game end. Draw')
                return winner

    def batch_self_play(self, player, n_games, temperature, temp_thres):
        envs = [self.env.copy() for _ in range(n_games)]
        for env in envs:
            env.reset()
            player.mcts.reset_env(envs.index(env))
        trajectories = [{'states': [], 'actions': [], 'probs': [], 'players': [], 'steps': 0}
                        for _ in range(n_games)]
        active_indices = list(range(n_games))
        completed_data = [None] * n_games
        while active_indices:
            current_boards = np.array([envs[i].board for i in range(n_games)])
            turns = np.array([envs[i].turn for i in range(n_games)], dtype=np.int32)
            temps = [temperature if trajectories[i]['steps'] <= temp_thres else 1e-3 for i in range(n_games)]
            actions, probs = player.get_action(current_boards, turns, temps)
            next_active_indices = []
            for i in active_indices:
                action = actions[i]
                prob = probs[i]
                env = envs[i]
                traj = trajectories[i]

                state_feature = env.current_state()[0].astype(np.int8)
                
                traj['states'].append(state_feature)
                traj['actions'].append(action)
                traj['probs'].append(prob)
                traj['players'].append(env.turn)
                traj['steps'] += 1

                env.step(action)
                if env.done():
                    winner = env.winPlayer()
                    
                    # 构建回放数据
                    states = traj['states']
                    # next_states 是 states 错位，最后补一个终止状态特征
                    end_state_feature = env.current_state()[0].astype(np.int8)
                    next_states = states[1:] + [end_state_feature]
                    
                    winner_z = np.zeros(len(traj['players']), dtype=np.int32)
                    if winner != 0:
                        winner_z[np.array(traj['players']) == winner] = 1
                        winner_z[np.array(traj['players']) != winner] = -1
                    
                    discount = reversed([pow(player.discount, k) for k in range(len(winner_z))])
                    dones = [False] * len(traj['players'])
                    dones[-1] = True
                    
                    # 此时 states 的 list 元素形状为 (3, 6, 7)，符合 dataset 预期
                    play_data = zip(states, traj['actions'], traj['probs'], discount, winner_z, next_states, dones)
                    completed_data[i] = (winner, tuple(play_data))
                    
                    # 游戏结束，重置该环境的 MCTS 树
                    player.mcts.reset_env(i)
                else:
                    next_active_indices.append(i)
            
            active_indices = next_active_indices

        return completed_data

    def streaming_self_play(self, player, n_games, temperature, temp_thres):
        """与 batch_self_play 语义相同，但 slot 游戏结束后立刻重置开新局，
        始终保持满 batch，避免 batch 末尾空洞浪费 MCTS 算力。
        返回恰好 n_games 局数据（不多不少）。"""
        batch_size = player.mcts.batch_size
        # 预计算最长游戏长度的 discount 向量，避免每局 pow 循环
        max_steps = self.env.board.size  # Connect4 = 42
        _disc_base = np.array([player.discount ** k for k in range(max_steps)], dtype=np.float32)

        envs = [self.env.copy() for _ in range(batch_size)]
        for i, env in enumerate(envs):
            env.reset()
            player.mcts.reset_env(i)

        def _new_traj():
            return {'states': [], 'actions': [], 'probs': [], 'players': [], 'steps': 0}

        trajectories = [_new_traj() for _ in range(batch_size)]
        completed_data = []

        while len(completed_data) < n_games:
            current_boards = np.array([env.board for env in envs])
            turns = np.array([env.turn for env in envs], dtype=np.int32)
            temps = [temperature if trajectories[i]['steps'] <= temp_thres else 1e-3
                     for i in range(batch_size)]
            actions, probs = player.get_action(current_boards, turns, temps)

            for i in range(batch_size):
                if len(completed_data) >= n_games:
                    break
                env = envs[i]
                traj = trajectories[i]

                traj['states'].append(env.current_state()[0].astype(np.int8))
                traj['actions'].append(actions[i])
                traj['probs'].append(probs[i])
                traj['players'].append(env.turn)
                traj['steps'] += 1

                env.step(actions[i])
                if env.done():
                    winner = env.winPlayer()
                    T = len(traj['players'])
                    states = traj['states']
                    next_states = states[1:] + [env.current_state()[0].astype(np.int8)]
                    winner_z = np.zeros(T, dtype=np.int32)
                    if winner != 0:
                        player_arr = np.array(traj['players'])
                        winner_z[player_arr == winner] = 1
                        winner_z[player_arr != winner] = -1
                    # 从预算好的向量直接切片并逆序，避免 pow 循环
                    discount = _disc_base[T - 1::-1]
                    dones = np.zeros(T, dtype=bool)
                    dones[-1] = True
                    play_data = tuple(zip(states, traj['actions'], traj['probs'], discount, winner_z, next_states, dones))
                    completed_data.append((winner, play_data))

                    # 立刻重置该 slot，开新局
                    env.reset()
                    player.mcts.reset_env(i)
                    trajectories[i] = _new_traj()

        return completed_data
