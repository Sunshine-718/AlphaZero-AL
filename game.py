import numpy as np


class Game:
    def __init__(self, env):
        self.env = env

    def start_self_play(self, player, temp=1, first_n_steps=5):
        self.env.reset()
        states, mcts_probs, current_players, next_states = [], [], [], []
        # values = []
        steps = 0
        while True:
            temperature = 1e-3 if steps >= first_n_steps else temp
            action, probs = player.get_action(self.env, temperature)
            # values.append(float(v_target))
            steps += 1
            states.append(self.env.current_state())
            mcts_probs.append(probs)
            current_players.append(self.env.turn)
            self.env.step(action)
            next_states.append(self.env.current_state())
            if self.env.done():
                winner = self.env.winPlayer()
                winner_z = np.zeros(len(current_players))
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1
                    winner_z[np.array(current_players) != winner] = -1
                dones = [False]*len(current_players)
                dones[-1] = True
                return winner, zip(states, mcts_probs, winner_z, winner_z, next_states, dones)
