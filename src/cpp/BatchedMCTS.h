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
    /**
     * 批量 MCTS 管理器 — 管理多局并行的搜索树。
     *
     * 每个环境（env）拥有独立的 MCTS 搜索树，通过 OpenMP 并行执行。
     * 工作流：
     *   1. search_batch() — 并行选择叶节点，返回待评估的棋盘状态
     *   2. Python 端批量 NN 推理
     *   3. backprop_batch() — 并行反向传播评估结果
     *   4. 重复 n_playout 次
     *
     * 对于纯 MCTS 基线，rollout_playout() 将整个循环留在 C++ 内。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class BatchedMCTS
    {
    private:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
        static constexpr int BOARD_SIZE = Game::Traits::BOARD_SIZE;

        int n_envs;                                             ///< 并行环境数量
        std::vector<std::unique_ptr<MCTS<Game>>> mcts_envs;     ///< 每局独立的 MCTS 搜索树

    public:
        /**
         * 构造批量 MCTS。
         * @param num_envs       并行环境数量
         * @param c_init         PUCT 初始常数
         * @param c_base         PUCT 对数基数
         * @param alpha          Dirichlet 噪声 alpha（≤0 禁用）
         * @param noise_epsilon  噪声混合权重
         * @param fpu_reduction  FPU 衰减系数
         * @param use_symmetry   是否启用随机对称增强
         * @param mlh_slope      MLH 斜率
         * @param mlh_cap        MLH 上限
         */
        BatchedMCTS(int num_envs, float c_init, float c_base, float alpha,
                    float noise_epsilon = 0.25f, float fpu_reduction = 0.4f, bool use_symmetry = true,
                    float mlh_slope = 0.0f, float mlh_cap = 0.2f)
            : n_envs(num_envs)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs.push_back(std::make_unique<MCTS<Game>>(c_init, c_base, alpha, noise_epsilon, fpu_reduction, use_symmetry,
                                                                  mlh_slope, mlh_cap));
            }
        }

        /**
         * 设置所有 OpenMP 线程的随机种子。
         * @param seed 种子值；负数时使用 std::random_device 重新随机化
         */
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

        /// 获取并行环境数量
        int get_num_envs() const { return n_envs; }

        /**
         * 重置指定环境的搜索树（清空节点池，创建新根节点）。
         * @param env_idx 环境索引
         */
        void reset_env(int env_idx)
        {
            if (env_idx >= 0 && env_idx < n_envs)
            {
                mcts_envs[env_idx]->reset();
            }
        }

        /**
         * 批量设置噪声混合权重 ε。
         * 训练时使用正值（如 0.25）增加探索；评估时设为 0 关闭噪声。
         * @param eps 噪声权重
         */
        void set_noise_epsilon(float eps)
        {
            for (auto &m : mcts_envs)
            {
                m->noise_epsilon = eps;
            }
        }

        /**
         * 批量设置 Moves Left Head 参数。
         * @param slope MLH 斜率（0 = 禁用）
         * @param cap   MLH 最大影响上限
         */
        void set_mlh_params(float slope, float cap)
        {
            for (auto &m : mcts_envs)
            {
                m->mlh_slope = slope;
                m->mlh_cap = cap;
            }
        }

        /**
         * 批量树剪枝：每局选择实际落子的动作，将对应子树提升为新根节点。
         * @param actions 每局的落子动作，长度 n_envs
         */
        void prune_roots(std::span<const int> actions)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs[i]->prune_root(actions[i]);
            }
        }

        /**
         * 批量 Selection 阶段：并行选择叶节点。
         *
         * 每局独立执行 MCTS::simulate()，返回叶节点状态供外部 NN 评估。
         *
         * @param input_boards       输入棋盘 (n_envs × BOARD_SIZE)，int8
         * @param turns              当前落子方 (n_envs,)，1 或 -1
         * @param output_boards      [输出] 叶节点棋盘 (n_envs × BOARD_SIZE)
         * @param output_term_values [输出] 终局价值 (n_envs,)
         * @param output_is_term     [输出] 是否终局 (n_envs,)
         * @param output_turns       [输出] 叶节点落子方 (n_envs,)
         */
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
                float term_val = 0.0f;

                mcts_envs[i]->simulate(current_game, leaf_board, is_term, term_val);

                output_is_term[i] = is_term ? 1 : 0;
                output_term_values[i] = term_val;
                output_turns[i] = leaf_board.get_turn();

                std::memcpy(output_boards + offset, leaf_board.board_data(), BOARD_SIZE);
            }
        }

        /**
         * 批量 Backpropagation 阶段：并行反向传播 NN 评估结果。
         *
         * @param policy_logits NN 策略输出 (n_envs × ACTION_SIZE)
         * @param values        NN 价值输出 (n_envs,)
         * @param moves_left    NN 剩余步数输出 (n_envs,)
         * @param is_term       是否终局 (n_envs,)
         */
        void backprop_batch(
            const float *policy_logits,
            const float *values,
            const float *moves_left,
            const uint8_t *is_term)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                float val = values[i];
                float ml = moves_left[i];

                std::array<float, ACTION_SIZE> policy;
                int offset = i * ACTION_SIZE;
                for (int a = 0; a < ACTION_SIZE; ++a)
                {
                    policy[a] = policy_logits[offset + a];
                }
                mcts_envs[i]->backprop(policy, val, ml, is_term[i] != 0);
            }
        }

        // ========== Random Rollout（纯 MCTS 基线）==========

        /**
         * 纯 MCTS playout：整个 n_playout 循环在 C++ 内完成，无需 Python 回调。
         *
         * 每次 playout 对所有环境并行执行 simulate + random_rollout + backprop。
         * 叶节点用 uniform prior 展开，价值由随机模拟到终局得到。
         *
         * @param input_boards 输入棋盘 (n_envs × BOARD_SIZE)，int8
         * @param turns        当前落子方 (n_envs,)
         * @param n_playout    模拟次数
         */
        void rollout_playout(const int8_t *input_boards, const int *turns, int n_playout)
        {
            for (int p = 0; p < n_playout; ++p)
            {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n_envs; ++i)
                {
                    Game current_game;
                    current_game.import_board(input_boards + i * BOARD_SIZE);
                    current_game.set_turn(turns[i]);
                    mcts_envs[i]->simulate_with_rollout(current_game);
                }
            }
        }

        /**
         * 获取所有环境根节点各动作的访问次数。
         * @return flat 向量 (n_envs × ACTION_SIZE)，counts[i*A + a] = 环境 i 动作 a 的访问次数
         */
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

        /// 每个环境的 root stats flat 长度：3 (root_N, root_Q, root_M) + ACTION_SIZE × 5
        static constexpr int STATS_PER_ENV = 3 + ACTION_SIZE * 5;

        /**
         * 获取所有环境的根节点统计量，写入 flat 缓冲区。
         *
         * 每个环境布局：[root_N, root_Q, root_M, a0_N, a0_Q, a0_prior, a0_noise, a0_M, ...]
         *
         * @param out 预分配缓冲区，大小 = n_envs × STATS_PER_ENV
         */
        void get_all_root_stats(float *out)
        {
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs[i]->get_root_stats(out + i * STATS_PER_ENV);
            }
        }
    };
}

#endif /* CD4397BA_1EEC_4275_BB07_6481002E649E */
