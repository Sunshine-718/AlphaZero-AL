#ifndef CD4397BA_1EEC_4275_BB07_6481002E649E
#define CD4397BA_1EEC_4275_BB07_6481002E649E
#pragma once
#include "MCTS.h"
#include <omp.h>
#include <span>
#include <vector>

namespace AlphaZero
{
    class BatchedMCTS
    {
    private:
        int n_envs;
        std::vector<std::unique_ptr<MCTS>> mcts_envs;

    public:
        BatchedMCTS(int num_envs, float c_init, float c_base, float discount, float alpha)
            : n_envs(num_envs)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs.push_back(std::make_unique<MCTS>(c_init, c_base, discount, alpha));
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
            int *output_turns) // [ROBUSTNESS FIX] 新增 output_turns
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                Connect4 current_game;
                int offset = i * Config::ROWS * Config::COLS;

                std::memcpy(current_game.board, input_boards + offset, Config::ROWS * Config::COLS);
                current_game.turn = turns[i];

                Connect4 leaf_board;
                bool is_term;
                float term_val;

                mcts_envs[i]->simulate(current_game, leaf_board, is_term, term_val);

                output_is_term[i] = is_term ? 1 : 0;
                output_term_values[i] = term_val;

                // [ROBUSTNESS FIX] 直接返回 C++ 状态中的轮次，不让 Python 猜
                output_turns[i] = leaf_board.turn;

                std::memcpy(output_boards + offset, leaf_board.board, Config::ROWS * Config::COLS);
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
                // 如果是终端状态，MCTS 应该使用之前 search_batch 算出的真实收益，
                // 或者是这里传递进来的值（假设 Python 端已经处理好）。
                // 通常 Python 端逻辑是：如果 is_term，val = term_val。
                // 我们直接信任传入的 values[i]。

                std::vector<float> policy(Config::ACTION_SIZE);
                int offset = i * Config::ACTION_SIZE;
                for (int a = 0; a < Config::ACTION_SIZE; ++a)
                {
                    policy[a] = policy_logits[offset + a];
                }
                mcts_envs[i]->backprop(policy, val);
            }
        }

        std::vector<int> get_all_counts()
        {
// #pragma omp parallel for schedule(static)
            std::vector<int> all_counts(n_envs * Config::ACTION_SIZE);

            for (int i = 0; i < n_envs; ++i)
            {
                std::vector<int> counts = mcts_envs[i]->get_counts();
                int offset = i * Config::ACTION_SIZE;
                for (int a = 0; a < Config::ACTION_SIZE; ++a)
                {
                    all_counts[offset + a] = counts[a];
                }
            }
            return all_counts;
        }
    };
}

#endif /* CD4397BA_1EEC_4275_BB07_6481002E649E */
