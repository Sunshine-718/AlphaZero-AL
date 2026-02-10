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

    def self_play(self, player, temp_discount=0.93):
        self.env.reset()
        states, actions, mcts_probs, current_players, next_states = [], [], [], [], []
        steps = 0
        while True:
            temperature = pow(temp_discount, steps)
            action, probs = player.get_action(self.env, temperature)
            steps += 1
            states.append(self.env.current_state().astype(np.int8))
            actions.append(action)
            mcts_probs.append(probs)
            current_players.append(self.env.turn)
            self.env.step(action)
            next_states.append(self.env.current_state().astype(np.int8))
            if self.env.done():
                winner = self.env.winPlayer()
                winner_z = np.zeros(len(current_players), dtype=np.int32)
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1
                    winner_z[np.array(current_players) != winner] = -1
                discount = reversed([pow(player.discount, i) for i in range(len(winner_z))])
                dones = [False] * len(current_players)
                dones[-1] = True
                return winner, zip(states, actions, mcts_probs, discount, winner_z, next_states, dones)
