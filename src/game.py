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

    @staticmethod
    def _get_temp(step, temp_init, temp_decay_moves, temp_endgame):
        """lc0-style linear temperature decay with floor.

        temp(step) = temp_init * max(0, 1 - step / decay_moves)
        clamped at temp_endgame from below.
        If decay_moves <= 0, always returns temp_init (no decay).
        """
        if temp_decay_moves <= 0:
            return temp_init
        t = temp_init * max(0.0, 1.0 - step / temp_decay_moves)
        return max(t, temp_endgame)

    def batch_self_play(self, player, n_games, temperature, temp_decay_moves, temp_endgame=0):
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
            temps = [self._get_temp(trajectories[i]['steps'], temperature, temp_decay_moves, temp_endgame)
                     for i in range(n_games)]
            # noise epsilon 衰减：开局高探索，随棋局线性衰减到 noise_eps_min
            if getattr(player, 'noise_steps', 0) > 0:
                step = trajectories[active_indices[0]]['steps']
                decay = max(0.0, 1.0 - step / player.noise_steps)
                eps = player.noise_eps_min + (player.noise_eps_init - player.noise_eps_min) * decay
                player.mcts.set_noise_epsilon(eps)
            actions, probs = player.get_batch_action(current_boards, turns, temps)
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
                    
                    T = len(traj['players'])
                    winner_z = np.full(T, winner, dtype=np.int32)
                    steps_to_end = np.arange(T, 0, -1, dtype=np.int32)

                    # 此时 states 的 list 元素形状为 (3, 6, 7)，符合 dataset 预期
                    play_data = list(zip(states, traj['probs'], winner_z, steps_to_end))
                    # 追加终局状态: steps_to_end=0, prob 全零
                    zero_prob = np.zeros_like(traj['probs'][0])
                    play_data.append((end_state_feature, zero_prob, winner, 0))
                    completed_data[i] = (winner, tuple(play_data))
                    
                    # 游戏结束，重置该环境的 MCTS 树
                    player.mcts.reset_env(i)
                else:
                    next_active_indices.append(i)
            
            active_indices = next_active_indices

        return completed_data

