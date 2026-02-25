#ifndef D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#define D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#pragma once
#include "GameContext.h"
#include "MCTSNode.h"
#include <random>
#include <span>
#include <vector>

namespace AlphaZero
{
    /// 获取线程本地的随机数生成器（OpenMP 并行安全）
    inline std::mt19937 &get_rng()
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        return rng;
    }

    /**
     * 单棵 MCTS 搜索树，对应一局游戏。
     *
     * 核心流程：simulate() 选择叶节点 → 外部评估（NN 或 rollout）→ backprop() 反向传播。
     * 节点以线性池（node_pool）管理，通过 int32_t 索引引用，避免指针开销。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class MCTS
    {
    public:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
        using Node = MCTSNode<ACTION_SIZE>;

        std::vector<Node> node_pool;        ///< 节点池（线性数组，索引访问）
        int32_t root_idx = 0;               ///< 当前根节点在 node_pool 中的索引
        int32_t next_free_node = 0;         ///< 下一个可分配的节点索引

        Game sim_env;                       ///< simulate() 中用于模拟走步的临时游戏状态
        Game leaf_state;                    ///< 最近一次 simulate() 到达的叶节点状态（未应用对称变换）
        int32_t current_leaf_idx = -1;      ///< 最近一次 simulate() 到达的叶节点索引
        int current_sym_id = 0;             ///< 最近一次 simulate() 使用的对称变换 ID

        float c_init, c_base, alpha;        ///< PUCT 常数、Dirichlet alpha
        float noise_epsilon;                ///< Dirichlet 噪声混合权重 ε
        float fpu_reduction;                ///< First Play Urgency 衰减系数
        bool use_symmetry;                  ///< 是否对叶节点随机应用对称变换
        float mlh_slope;                    ///< Moves Left Head 斜率
        float mlh_cap;                      ///< Moves Left Head 最大影响上限

        /**
         * 构造 MCTS 搜索树。
         * @param c_i       PUCT 初始常数 c_init
         * @param c_b       PUCT 对数基数 c_base
         * @param a         Dirichlet 噪声 alpha（≤0 时禁用噪声）
         * @param noise_eps 噪声混合权重 ε
         * @param fpu_red   FPU 衰减系数
         * @param use_sym   是否启用随机对称增强
         * @param mlh_slope_ MLH 斜率
         * @param mlh_cap_   MLH 上限
         */
        MCTS(float c_i, float c_b, float a, float noise_eps = 0.25f, float fpu_red = 0.4f, bool use_sym = true,
             float mlh_slope_ = 0.0f, float mlh_cap_ = 0.2f)
            : c_init(c_i), c_base(c_b), alpha(a), noise_epsilon(noise_eps), fpu_reduction(fpu_red), use_symmetry(use_sym),
              mlh_slope(mlh_slope_), mlh_cap(mlh_cap_)
        {
            node_pool.resize(2000);
            reset();
        }

        /// 重置搜索树：清空节点池，创建新根节点
        void reset()
        {
            next_free_node = 0;
            root_idx = allocate_node(-1, 1.0f);
        }

        /**
         * 从节点池分配一个新节点。
         * 池空间不足时自动 2 倍扩容。
         * @param parent 父节点索引
         * @param prior  策略先验概率
         * @return 新节点在 node_pool 中的索引
         */
        int32_t allocate_node(int32_t parent, float prior)
        {
            if (next_free_node >= static_cast<int32_t>(node_pool.size()))
            {
                node_pool.resize(node_pool.size() * 2);
            }
            int32_t idx = next_free_node++;
            node_pool[idx].reset(parent, prior);
            return idx;
        }

        /**
         * 树剪枝：落子 action 后，将对应子节点提升为新根节点。
         * 如果子节点不存在（action 无效），则重置整棵树。
         * 剪枝后为新根节点重新生成 Dirichlet 噪声。
         * @param action 实际落子的动作
         */
        void prune_root(int action)
        {
            int32_t child_idx = node_pool[root_idx].children[action];
            if (action >= 0 && action < ACTION_SIZE && child_idx != -1)
            {
                root_idx = child_idx;
                node_pool[root_idx].parent = -1;
                apply_root_noise();
            } else
            {
                reset();
            }
        }

        /**
         * 为当前根节点的所有已有子节点重新生成 Dirichlet 噪声。
         * 仅在 alpha > 0 且根节点已展开时生效。
         * 通过 Gamma 分布采样后归一化得到 Dirichlet 分布。
         */
        void apply_root_noise()
        {
            if (alpha <= 0.0f || !node_pool[root_idx].is_expanded) return;

            std::mt19937 &rng = get_rng();
            std::gamma_distribution<float> gamma(alpha, 1.0f);

            int32_t valid_children[ACTION_SIZE];
            int n_valid = 0;
            for (int a = 0; a < ACTION_SIZE; ++a)
            {
                if (node_pool[root_idx].children[a] != -1)
                {
                    valid_children[n_valid++] = node_pool[root_idx].children[a];
                }
            }
            if (n_valid == 0) return;

            float noise_arr[ACTION_SIZE];
            float sum = 0.0f;
            for (int i = 0; i < n_valid; ++i)
            {
                noise_arr[i] = gamma(rng);
                sum += noise_arr[i];
            }
            for (int i = 0; i < n_valid; ++i)
            {
                node_pool[valid_children[i]].noise = noise_arr[i] / sum;
            }
        }

        /**
         * MCTS 模拟阶段（Selection）：从根节点沿搜索树向下选择，直到到达叶节点或终局状态。
         *
         * 在已展开的节点中，按 UCB 分数选择最优动作并前进；
         * 遇到未展开节点（叶节点）或终局状态时停止。
         *
         * 对于非终局叶节点，若启用对称增强，会随机对叶节点状态应用对称变换，
         * 以增加神经网络的输入多样性（逆变换在 backprop 中恢复）。
         *
         * @param start_state        当前棋盘状态
         * @param out_nn_input_board [输出] 叶节点状态（可能经过对称变换），用于 NN 评估
         * @param out_is_terminal    [输出] 是否为终局状态
         * @param out_terminal_val   [输出] 终局价值（-1=上一手赢, 0=平局），仅终局有效
         */
        void simulate(const Game &start_state, Game &out_nn_input_board, bool &out_is_terminal, float &out_terminal_val)
        {
            sim_env = start_state;
            int32_t curr_idx = root_idx;
            current_sym_id = 0;

            // Selection：沿已展开路径选择最优动作
            while (node_pool[curr_idx].is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float p_n = static_cast<float>(node_pool[curr_idx].n_visits);

                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

                // 计算 FPU（First Play Urgency）：未访问过的动作的默认价值估计
                // 劣势时降低 FPU，鼓励探索新走法
                float parent_value = node_pool[curr_idx].Q;
                float seen_policy = 0.0f;
                for (int action : valids)
                {
                    int32_t child_idx = node_pool[curr_idx].children[action];
                    if (child_idx != -1 && node_pool[child_idx].n_visits > 0)
                    {
                        seen_policy += node_pool[child_idx].prior;
                    }
                }
                float scale = (1.0f + parent_value) / 2.0f;
                float effective_fpu = fpu_reduction * scale;
                float fpu_value = parent_value - effective_fpu * std::sqrt(seen_policy);
                fpu_value = std::max(-1.0f, fpu_value);

                // 对每个合法动作计算 UCB，选分数最高的
                float parent_M = node_pool[curr_idx].M;
                for (int action : valids)
                {
                    int32_t child_idx = node_pool[curr_idx].children[action];
                    if (child_idx != -1)
                    {
                        float score = node_pool[child_idx].get_ucb(c_init, c_base, p_n, curr_idx == root_idx, noise_epsilon, fpu_value,
                                                                    parent_M, mlh_slope, mlh_cap);
                        if (score > best_score)
                        {
                            best_score = score;
                            best_action = action;
                        }
                    }
                }
                if (best_action == -1) break;

                sim_env.step(best_action);
                curr_idx = node_pool[curr_idx].children[best_action];

                if (sim_env.check_winner() != 0 || sim_env.is_full()) break;
            }
            current_leaf_idx = curr_idx;

            // 检查是否为终局状态
            int winner = sim_env.check_winner();
            if (winner != 0)
            {
                out_is_terminal = true;
                out_terminal_val = -1.0f;       // 有人赢了 → 对当前落子方来说是 -1
                out_nn_input_board = sim_env;
                return;
            } else if (sim_env.is_full())
            {
                out_is_terminal = true;
                out_terminal_val = 0.0f;        // 平局
                out_nn_input_board = sim_env;
                return;
            }

            // 非终局叶节点：保存原始状态，可选应用随机对称变换
            out_is_terminal = false;
            leaf_state = sim_env;

            if (use_symmetry && Game::Traits::NUM_SYMMETRIES > 1)
            {
                std::mt19937 &rng = get_rng();
                std::uniform_int_distribution<int> sym_dist(0, Game::Traits::NUM_SYMMETRIES - 1);
                current_sym_id = sym_dist(rng);
                sim_env.apply_symmetry(current_sym_id);
            }

            out_nn_input_board = sim_env;
        }

        /**
         * 反向传播阶段（Expansion + Backpropagation）：用评估结果展开叶节点并更新路径上的统计量。
         *
         * 非终局叶节点：
         *   1. 对 policy 应用对称逆变换（恢复 simulate 中的随机对称）
         *   2. 在根节点处生成 Dirichlet 噪声
         *   3. 为所有合法动作创建子节点，设置策略先验
         * 终局叶节点：不展开，仅反向传播。
         *
         * 反向传播：从叶节点沿 parent 链向上更新 N, Q, M，每层翻转 value 符号。
         *
         * @param policy_logits NN 输出的策略 logits（或 uniform），长度 ACTION_SIZE
         * @param value         叶节点评估值 ∈ [-1, 1]
         * @param moves_left    预期剩余步数（NN 输出，纯 MCTS 时为 0）
         * @param is_terminal   是否为终局状态
         */
        void backprop(std::span<const float> policy_logits, float value, float moves_left, bool is_terminal)
        {
            if (current_leaf_idx == -1) return;

            // 终局状态不展开，直接反向传播
            if (!is_terminal)
            {
                // 对称逆变换：恢复 simulate 中的随机对称
                std::array<float, ACTION_SIZE> final_policy;
                std::copy(policy_logits.begin(), policy_logits.end(), final_policy.begin());
                Game::inverse_symmetry_policy(current_sym_id, final_policy);

                auto valids = leaf_state.get_valid_moves();

                // 根节点首次展开时生成 Dirichlet 噪声
                float noise_arr[ACTION_SIZE];
                int noise_count = 0;
                if (node_pool[current_leaf_idx].parent == -1 && alpha > 0.0f)
                {
                    std::mt19937 &rng = get_rng();
                    std::gamma_distribution<float> gamma(alpha, 1.0f);

                    float sum = 0.0f;
                    for (int i = 0; i < valids.size(); ++i)
                    {
                        float n = gamma(rng);
                        noise_arr[i] = n;
                        sum += n;
                    }
                    noise_count = valids.size();
                    for (int i = 0; i < noise_count; ++i)
                        noise_arr[i] /= sum;
                }

                // 策略归一化：只对合法动作的 logits 求和
                float policy_sum = 0.0f;
                for (int action : valids)
                {
                    policy_sum += final_policy[action];
                }

                // 展开：为每个合法动作创建子节点
                int noise_idx = 0;
                for (int action : valids)
                {
                    float prob = final_policy[action] / (policy_sum + 1e-8f);
                    if (node_pool[current_leaf_idx].children[action] == -1)
                    {
                        int32_t new_node = allocate_node(current_leaf_idx, prob);
                        node_pool[current_leaf_idx].children[action] = new_node;

                        if (noise_count > 0)
                        {
                            node_pool[new_node].noise = noise_arr[noise_idx++];
                        }
                    }
                }
                node_pool[current_leaf_idx].is_expanded = true;
            }

            // 反向传播：从叶节点沿 parent 链向上更新统计量
            // 每层翻转 value（对手视角），moves_left 递增 1（父节点比子节点多一步）
            int32_t update_idx = current_leaf_idx;
            float val = value;
            float ml = is_terminal ? 0.0f : moves_left;
            while (update_idx != -1)
            {
                node_pool[update_idx].n_visits++;
                node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
                node_pool[update_idx].M += (ml - node_pool[update_idx].M) / node_pool[update_idx].n_visits;
                val = -val;
                ml += 1.0f;
                update_idx = node_pool[update_idx].parent;
            }
        }

        // ========== Random Rollout（纯 MCTS 基线）==========

        /**
         * 随机模拟（rollout）：从给定状态随机落子直到终局。
         * 用于纯 MCTS 基线的叶节点评估，替代神经网络。
         *
         * @param state 起始游戏状态（按值传递，内部修改不影响外部）
         * @return 终局价值：-1（上一手赢了）或 0（平局），相对于「即将落子方」
         */
        float random_rollout(Game state) const
        {
            std::mt19937 &rng = get_rng();
            while (true)
            {
                if (state.check_winner() != 0) return -1.0f;
                if (state.is_full()) return 0.0f;
                auto valids = state.get_valid_moves();
                std::uniform_int_distribution<int> dist(0, valids.size() - 1);
                state.step(valids.moves[dist(rng)]);
            }
        }

        /**
         * 一步完成 simulate + rollout + backprop，无需 Python 回调。
         * 用于纯 MCTS 基线：对叶节点用 random_rollout 评估，用 uniform prior 展开。
         * @param start_state 当前棋盘状态
         */
        void simulate_with_rollout(const Game &start_state)
        {
            Game out_board;
            bool is_terminal;
            float term_val;
            simulate(start_state, out_board, is_terminal, term_val);

            if (is_terminal)
            {
                std::array<float, ACTION_SIZE> dummy{};
                backprop(dummy, term_val, 0.0f, true);
            }
            else
            {
                float val = random_rollout(leaf_state);
                std::array<float, ACTION_SIZE> uniform;
                uniform.fill(1.0f);
                backprop(uniform, val, 0.0f, false);
            }
        }

        /**
         * 获取根节点各动作的访问次数。
         * @return 长度为 ACTION_SIZE 的向量，counts[a] = 动作 a 的子节点访问次数
         */
        std::vector<int> get_counts() const
        {
            std::vector<int> counts(ACTION_SIZE, 0);
            for (int i = 0; i < ACTION_SIZE; ++i)
            {
                int32_t child_idx = node_pool[root_idx].children[i];
                if (child_idx != -1)
                {
                    counts[i] = node_pool[child_idx].n_visits;
                }
            }
            return counts;
        }

        /**
         * 将根节点及其子节点的统计量写入 flat 缓冲区，供 Python 端读取。
         *
         * 布局：[root_N, root_Q, root_M, a0_N, a0_Q, a0_prior, a0_noise, a0_M, a1_N, ...]
         * 总长度 = 3 + ACTION_SIZE × 5
         *
         * @param out 预分配的输出缓冲区
         */
        void get_root_stats(float *out) const
        {
            const Node &root = node_pool[root_idx];
            out[0] = static_cast<float>(root.n_visits);
            out[1] = root.Q;
            out[2] = root.M;

            float *p = out + 3;
            for (int a = 0; a < ACTION_SIZE; ++a)
            {
                int32_t child_idx = root.children[a];
                if (child_idx != -1)
                {
                    const Node &child = node_pool[child_idx];
                    p[0] = static_cast<float>(child.n_visits);
                    p[1] = child.Q;
                    p[2] = child.prior;
                    p[3] = child.noise;
                    p[4] = child.M;
                }
                else
                {
                    p[0] = p[1] = p[2] = p[3] = p[4] = 0.0f;
                }
                p += 5;
            }
        }
    };
}

#endif /* D77F1FBF_0C41_461C_8809_93FD96ACA0C5 */
