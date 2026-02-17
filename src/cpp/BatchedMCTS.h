#ifndef CD4397BA_1EEC_4275_BB07_6481002E649E
#define CD4397BA_1EEC_4275_BB07_6481002E649E
#pragma once
#include "MCTS.h"
#include <array>
#include <omp.h>
#include <span>
#include <vector>

namespace AlphaZero
{
    template <MCTSGame Game>
    class BatchedMCTS
    {
    private:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
        static constexpr int BOARD_SIZE = Game::Traits::BOARD_SIZE;

        int n_envs;
        std::vector<std::unique_ptr<MCTS<Game>>> mcts_envs;

    public:
        BatchedMCTS(int num_envs, float c_init, float c_base, float discount, float alpha,
                    float noise_epsilon = 0.25f)
            : n_envs(num_envs)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs.push_back(std::make_unique<MCTS<Game>>(c_init, c_base, discount, alpha, noise_epsilon));
            }
        }

        void set_seed(int seed)
        {
#pragma omp parallel
            {
                std::mt19937 &rng = get_rng();
                if (seed < 0)
                {
                    rng.seed(std::random_device{}());
                }
                else
                {
                    int thread_id = omp_get_thread_num();
                    unsigned int local_seed = static_cast<unsigned int>(seed + thread_id * 10007);
                    get_rng().seed(local_seed);
                }
            }
        }

        int get_num_envs() const { return n_envs; }

        void reset_env(int env_idx)
        {
            if (env_idx >= 0 && env_idx < n_envs)
            {
                mcts_envs[env_idx]->reset();
            }
        }

        void prune_roots(std::span<const int> actions)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs[i]->prune_root(actions[i]);
            }
        }

        void search_batch(
            const int8_t *input_boards,
            const int *turns,
            int8_t *output_boards,
            float *output_term_values,
            uint8_t *output_is_term,
            int *output_turns)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                Game current_game;
                int offset = i * BOARD_SIZE;

                current_game.import_board(input_boards + offset);
                current_game.set_turn(turns[i]);

                Game leaf_board;
                bool is_term;
                float term_val;

                mcts_envs[i]->simulate(current_game, leaf_board, is_term, term_val);

                output_is_term[i] = is_term ? 1 : 0;
                output_term_values[i] = term_val;
                output_turns[i] = leaf_board.get_turn();

                std::memcpy(output_boards + offset, leaf_board.board_data(), BOARD_SIZE);
            }
        }

        void backprop_batch(
            const float *policy_logits,
            const float *values,
            const uint8_t *is_term)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                float val = values[i];

                std::array<float, ACTION_SIZE> policy;
                int offset = i * ACTION_SIZE;
                for (int a = 0; a < ACTION_SIZE; ++a)
                {
                    policy[a] = policy_logits[offset + a];
                }
                mcts_envs[i]->backprop(policy, val, is_term[i] != 0);
            }
        }

        std::vector<int> get_all_counts()
        {
            std::vector<int> all_counts(n_envs * ACTION_SIZE);

            for (int i = 0; i < n_envs; ++i)
            {
                std::vector<int> counts = mcts_envs[i]->get_counts();
                int offset = i * ACTION_SIZE;
                for (int a = 0; a < ACTION_SIZE; ++a)
                {
                    all_counts[offset + a] = counts[a];
                }
            }
            return all_counts;
        }
    };
}

#endif /* CD4397BA_1EEC_4275_BB07_6481002E649E */
