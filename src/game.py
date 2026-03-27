#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:55
import numpy as np


class Game:
    def __init__(self, env):
        self.env = env
        mod = env.__class__.__module__.lower()
        if 'othello' in mod:
            self.aux_mode = 'terminal_disc_diff'
        else:
            self.aux_mode = 'steps_to_end'

    def _make_aux_targets(self, traj_players, terminal_env, steps_to_end):
        if self.aux_mode == 'terminal_disc_diff':
            board = np.asarray(terminal_env.board)
            diff = int(np.sum(board == 1) - np.sum(board == -1))
            players = np.asarray(traj_players, dtype=np.int32)
            return diff * players
        return steps_to_end

    def _terminal_aux_target(self, terminal_env):
        if self.aux_mode == 'terminal_disc_diff':
            board = np.asarray(terminal_env.board)
            diff = int(np.sum(board == 1) - np.sum(board == -1))
            return diff * int(terminal_env.turn)
        return 0

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

    def batch_self_play(self, player, n_games, temperature, temp_decay_moves, temp_endgame=0,
                        td_steps=0):
        envs = [self.env.copy() for _ in range(n_games)]
        for env in envs:
            env.reset()
            player.mcts.reset_env(envs.index(env))
        trajectories = [{'states': [], 'actions': [], 'probs': [], 'players': [], 'root_wdls': [],
                         'valid_masks': [], 'steps': 0}
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
            actions, probs, root_wdls = player.get_batch_action(current_boards, turns, temps)
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
                traj['root_wdls'].append(root_wdls[i])
                traj['valid_masks'].append(np.array(env.valid_mask(), dtype=np.bool_))
                traj['players'].append(env.turn)
                traj['steps'] += 1

                env.step(action)
                if env.done():
                    winner = env.winPlayer()

                    # 构建回放数据
                    states = traj['states']
                    end_state_feature = env.current_state()[0].astype(np.int8)

                    T = len(traj['players'])
                    winner_z = np.full(T, winner, dtype=np.int32)
                    steps_to_end = np.arange(T, 0, -1, dtype=np.int32)
                    aux_targets = self._make_aux_targets(traj['players'], env, steps_to_end)

                    # 构建 future_root_wdl: S_{t+k} 的根节点 WDL（绝对视角）
                    k = td_steps
                    zero_wdl = np.zeros(3, dtype=np.float32)
                    if k > 0:
                        future_root_wdls = []
                        for t in range(T):
                            ft = t + k
                            if ft < T:
                                future_root_wdls.append(traj['root_wdls'][ft])
                            else:
                                future_root_wdls.append(zero_wdl)
                        play_data = list(zip(states, traj['probs'], winner_z,
                                             steps_to_end, aux_targets, traj['root_wdls'],
                                             traj['valid_masks'], future_root_wdls))
                    else:
                        play_data = list(zip(states, traj['probs'], winner_z,
                                             steps_to_end, aux_targets, traj['root_wdls'],
                                             traj['valid_masks']))

                    # 追加终局状态: steps_to_end=0, prob 全零, root_wdl 全零
                    zero_prob = np.zeros_like(traj['probs'][0])
                    zero_mask = np.ones_like(traj['valid_masks'][0])
                    terminal_tuple = [end_state_feature, zero_prob, winner, 0,
                                      self._terminal_aux_target(env), zero_wdl,
                                      zero_mask]
                    if k > 0:
                        terminal_tuple.append(zero_wdl)
                    play_data.append(tuple(terminal_tuple))
                    completed_data[i] = (winner, tuple(play_data))
                    
                    # 游戏结束，重置该环境的 MCTS 树
                    player.mcts.reset_env(i)
                else:
                    next_active_indices.append(i)
            
            active_indices = next_active_indices

        return completed_data

