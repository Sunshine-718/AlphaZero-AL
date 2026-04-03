#pragma once
#include "MCTS.h"
#include "IEvaluator.h"
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
     * 对于纯 MCTS 基线，search() 将整个循环留在 C++ 内。
     * SearchConfig 由 BatchedMCTS 拥有，所有子树通过 const 指针共享。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class BatchedMCTS
    {
    private:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
        static constexpr int BOARD_SIZE = Game::Traits::BOARD_SIZE;

        static int sample_symmetry_id(std::mt19937 &rng)
        {
            if constexpr (requires(std::mt19937 &gen) { Game::sample_mcts_symmetry_id(gen); })
            {
                return Game::sample_mcts_symmetry_id(rng);
            }
            return std::uniform_int_distribution<int>(0, Game::Traits::NUM_SYMMETRIES - 1)(rng);
        }

        int n_envs;                                             ///< 并行环境数量
        SearchConfig config_;                                   ///< 搜索配置（本类拥有）
        std::vector<std::unique_ptr<MCTS<Game>>> mcts_envs;     ///< 每局独立的 MCTS 搜索树
        std::vector<int> pending_sym_ids_;                      ///< 每个 env 的待处理对称 ID

    public:
        /**
         * 构造批量 MCTS。
         * @param num_envs 并行环境数量
         */
        explicit BatchedMCTS(int num_envs)
            : n_envs(num_envs), pending_sym_ids_(num_envs, 0)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
                mcts_envs.push_back(std::make_unique<MCTS<Game>>(config_));
        }

        /// 获取搜索配置（可修改）
        SearchConfig& config() { return config_; }
        const SearchConfig& config() const { return config_; }

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
         */
        void search_batch(
            const int8_t *input_boards,
            const int *turns,
            int8_t *output_boards,
            float *output_term_d,
            float *output_term_p1w,
            float *output_term_p2w,
            uint8_t *output_is_term,
            int *output_turns,
            uint8_t *output_valid_mask)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                Game current_game;
                int offset = i * BOARD_SIZE;

                current_game.import_board(input_boards + offset);
                current_game.set_turn(turns[i]);

                auto result = mcts_envs[i]->simulate(current_game);

                output_is_term[i] = result.is_terminal ? 1 : 0;
                output_term_d[i] = result.terminal_wdl.d;
                output_term_p1w[i] = result.terminal_wdl.p1w;
                output_term_p2w[i] = result.terminal_wdl.p2w;
                output_turns[i] = result.board.get_turn();

                // 对非终局叶节点应用随机对称变换（增加 NN 输入多样性）
                if (!result.is_terminal && config_.use_symmetry && Game::Traits::NUM_SYMMETRIES > 1)
                {
                    std::mt19937 &rng = get_rng();
                    int sym_id = sample_symmetry_id(rng);
                    pending_sym_ids_[i] = sym_id;
                    if (sym_id != 0) result.board.apply_symmetry(sym_id);
                }
                else
                {
                    pending_sym_ids_[i] = 0;
                }

                std::memcpy(output_boards + offset, result.board.board_data(), BOARD_SIZE);

                uint8_t *mask_ptr = output_valid_mask + i * ACTION_SIZE;
                std::fill(mask_ptr, mask_ptr + ACTION_SIZE, 0);
                if (!result.is_terminal)
                {
                    auto valids = result.board.get_valid_moves();
                    for (int action : valids)
                        mask_ptr[action] = 1;
                }
            }
        }

        /**
         * 批量 Backpropagation 阶段：并行反向传播 NN 评估结果。
         */
        void backprop_batch(
            const float *policy_logits,
            const float *d_vals,
            const float *p1w_vals,
            const float *p2w_vals,
            const float *moves_left,
            const uint8_t *is_term,
            const float *ownership_occ = nullptr,
            const float *ownership_p1p2 = nullptr)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                std::array<float, ACTION_SIZE> policy;
                int offset = i * ACTION_SIZE;
                for (int a = 0; a < ACTION_SIZE; ++a)
                {
                    policy[a] = policy_logits[offset + a];
                }
                // 对称逆变换：恢复 search_batch 中的随机对称
                if (pending_sym_ids_[i] != 0)
                    Game::inverse_symmetry_policy(pending_sym_ids_[i], policy);
                WDLValue wdl{d_vals[i], p1w_vals[i], p2w_vals[i]};
                const float *occ_ptr = ownership_occ ? (ownership_occ + i * BOARD_SIZE) : nullptr;
                const float *p1p2_ptr = ownership_p1p2 ? (ownership_p1p2 + i * BOARD_SIZE) : nullptr;
                mcts_envs[i]->backprop(policy, wdl, moves_left[i], is_term[i] != 0,
                                       occ_ptr, p1p2_ptr);
            }
        }

        // ========== Virtual Loss 批量方法 ==========

        /**
         * 移除所有环境的 VL (n_inflight)。
         * Python 异常安全：当 NN 推理在 search_batch_vl 和 backprop_batch_vl 之间
         * 抛出异常时，调用此方法清除残留的 n_inflight。
         * @param K VL 模拟次数（与 search_batch_vl 一致）
         */
        void remove_all_vl(int K)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs[i]->remove_all_vl(K);
            }
        }

        /**
         * VL 批量 Selection：每棵树执行 K 次 VL 模拟，收集 N*K 个叶节点。
         *
         * 输出数组布局：[env0_leaf0, env0_leaf1, ..., env0_leafK-1, env1_leaf0, ...]
         * sym_ids 作为显式输出数组（大小 N*K），因为 pending_sym_ids_ 仅 N 大小。
         *
         * @param K       每棵树每次迭代的 VL 模拟次数
         * @param sym_ids 输出：对称变换 ID（N*K 长度）
         */
        void search_batch_vl(
            int K,
            const int8_t *input_boards,
            const int *turns,
            int8_t *output_boards,
            float *output_term_d,
            float *output_term_p1w,
            float *output_term_p2w,
            uint8_t *output_is_term,
            int *output_turns,
            int *sym_ids,
            uint8_t *output_valid_mask)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                Game current_game;
                current_game.import_board(input_boards + i * BOARD_SIZE);
                current_game.set_turn(turns[i]);

                mcts_envs[i]->prepare_vl(K);

                for (int k = 0; k < K; ++k)
                {
                    int flat_idx = i * K + k;
                    auto result = mcts_envs[i]->simulate_vl(k, current_game);

                    output_is_term[flat_idx] = result.is_terminal ? 1 : 0;
                    output_term_d[flat_idx] = result.terminal_wdl.d;
                    output_term_p1w[flat_idx] = result.terminal_wdl.p1w;
                    output_term_p2w[flat_idx] = result.terminal_wdl.p2w;
                    output_turns[flat_idx] = result.board.get_turn();

                    // 对称增强：每个叶节点独立随机变换
                    if (!result.is_terminal && config_.use_symmetry && Game::Traits::NUM_SYMMETRIES > 1)
                    {
                        std::mt19937 &rng = get_rng();
                        int sym_id = sample_symmetry_id(rng);
                        sym_ids[flat_idx] = sym_id;
                        if (sym_id != 0) result.board.apply_symmetry(sym_id);
                    }
                    else
                    {
                        sym_ids[flat_idx] = 0;
                    }

                    std::memcpy(output_boards + flat_idx * BOARD_SIZE,
                                result.board.board_data(), BOARD_SIZE);

                    uint8_t *mask_ptr = output_valid_mask + flat_idx * ACTION_SIZE;
                    std::fill(mask_ptr, mask_ptr + ACTION_SIZE, 0);
                    if (!result.is_terminal)
                    {
                        auto valids = result.board.get_valid_moves();
                        for (int action : valids)
                            mask_ptr[action] = 1;
                    }
                }
            }
        }

        /**
         * VL 批量 Backpropagation：先移除所有 VL，再逐个反向传播 K 个结果。
         *
         * 输入数组布局与 search_batch_vl 一致：[env0_leaf0, ..., env0_leafK-1, env1_leaf0, ...]
         *
         * @param K       VL 模拟次数（与 search_batch_vl 一致）
         * @param sym_ids 对称变换 ID（来自 search_batch_vl 的输出）
         */
        void backprop_batch_vl(
            int K,
            const float *policy_logits,
            const float *d_vals,
            const float *p1w_vals,
            const float *p2w_vals,
            const float *moves_left,
            const uint8_t *is_term,
            const int *sym_ids,
            const float *ownership_occ = nullptr,
            const float *ownership_p1p2 = nullptr)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_envs; ++i)
            {
                // Phase 1: 移除所有 VL，恢复树到干净状态
                mcts_envs[i]->remove_all_vl(K);

                // Phase 2: 逐个反向传播 K 个结果
                for (int k = 0; k < K; ++k)
                {
                    int flat_idx = i * K + k;

                    std::array<float, ACTION_SIZE> policy;
                    int p_offset = flat_idx * ACTION_SIZE;
                    for (int a = 0; a < ACTION_SIZE; ++a)
                        policy[a] = policy_logits[p_offset + a];

                    // 对称逆变换
                    if (sym_ids[flat_idx] != 0)
                        Game::inverse_symmetry_policy(sym_ids[flat_idx], policy);

                    WDLValue wdl{d_vals[flat_idx], p1w_vals[flat_idx], p2w_vals[flat_idx]};
                    const float *occ_ptr = ownership_occ ? (ownership_occ + flat_idx * BOARD_SIZE) : nullptr;
                    const float *p1p2_ptr = ownership_p1p2 ? (ownership_p1p2 + flat_idx * BOARD_SIZE) : nullptr;
                    mcts_envs[i]->backprop_vl(k, policy, wdl,
                                               moves_left[flat_idx],
                                               is_term[flat_idx] != 0,
                                               occ_ptr, p1p2_ptr);
                }
            }
        }

        // ========== 通用搜索入口（IEvaluator）==========

        /**
         * 通用搜索入口：用 IEvaluator 完成整个 playout 循环。
         */
        void search(IEvaluator<Game> &evaluator,
                    const int8_t *input_boards, const int *turns, int n_playout)
        {
            // 预分配 batch buffers
            std::vector<Game> leaf_states(n_envs);
            std::vector<int> leaf_turns(n_envs);
            std::vector<typename IEvaluator<Game>::Result> eval_results(n_envs);
            std::vector<bool> is_terminal(n_envs);

            for (int p = 0; p < n_playout; ++p)
            {
                // Phase 1: 并行 selection
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n_envs; ++i)
                {
                    Game current_game;
                    current_game.import_board(input_boards + i * BOARD_SIZE);
                    current_game.set_turn(turns[i]);
                    auto result = mcts_envs[i]->simulate(current_game);

                    is_terminal[i] = result.is_terminal;
                    if (result.is_terminal)
                    {
                        eval_results[i].policy.fill(0.0f);
                        eval_results[i].wdl = result.terminal_wdl;
                        eval_results[i].moves_left = 0.0f;
                    }
                    else
                    {
                        leaf_states[i] = result.board;
                        leaf_turns[i] = result.board.get_turn();
                    }
                }

                // Phase 2: 收集非终局叶节点，批量评估
                std::vector<int> non_term_indices;
                non_term_indices.reserve(n_envs);
                for (int i = 0; i < n_envs; ++i)
                    if (!is_terminal[i])
                        non_term_indices.push_back(i);

                if (!non_term_indices.empty())
                {
                    int n_eval = static_cast<int>(non_term_indices.size());
                    std::vector<Game> eval_states(n_eval);
                    std::vector<int> eval_turns(n_eval);
                    std::vector<typename IEvaluator<Game>::Result> batch_results(n_eval);

                    for (int j = 0; j < n_eval; ++j)
                    {
                        eval_states[j] = leaf_states[non_term_indices[j]];
                        eval_turns[j] = leaf_turns[non_term_indices[j]];
                    }

                    evaluator.evaluate_batch(eval_states, eval_turns, batch_results);

                    for (int j = 0; j < n_eval; ++j)
                        eval_results[non_term_indices[j]] = batch_results[j];
                }

                // Phase 3: 并行 backprop
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n_envs; ++i)
                {
                    auto &r = eval_results[i];
                    mcts_envs[i]->backprop(r.policy, r.wdl, r.moves_left, is_terminal[i]);
                }
            }
        }

        /**
         * 获取所有环境根节点各动作的访问次数。
         * @return flat 向量 (n_envs × ACTION_SIZE)
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

        /// 每个环境的 root stats flat 长度
        static constexpr int STATS_PER_ENV = 6 + ACTION_SIZE * 8;
        static constexpr bool HAS_OWNERSHIP = requires(const Game &g, float *occ, float *p1p2)
        {
            g.fill_absolute_ownership(occ, p1p2);
        };

        /**
         * 获取所有环境的根节点统计量，写入 flat 缓冲区。
         */
        void get_all_root_stats(float *out)
        {
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs[i]->get_root_stats(out + i * STATS_PER_ENV);
            }
        }

        void get_all_root_ownership(float *out)
        {
            if constexpr (HAS_OWNERSHIP)
            {
                constexpr int OWN_FLOATS = BOARD_SIZE * 3;
                for (int i = 0; i < n_envs; ++i)
                    mcts_envs[i]->get_root_ownership(out + i * OWN_FLOATS);
            }
        }
    };
}
