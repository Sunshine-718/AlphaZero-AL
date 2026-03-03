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
     * simulate() 的返回值：叶节点状态及其终局信息。
     * 使用结构体返回替代 4 个 out 参数，利用 NRVO 零开销。
     */
    template <MCTSGame Game>
    struct SimResult
    {
        Game board;             ///< 叶节点棋盘（可能经过对称变换）
        WDLValue terminal_wdl;  ///< 终局 WDL（绝对视角），仅终局有效
        bool is_terminal;       ///< 是否为终局状态
    };

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
        int32_t current_leaf_idx = -1;      ///< 最近一次 simulate() 到达的叶节点索引
        int current_leaf_turn = 1;          ///< 最近一次 simulate() 叶节点的落子方（1 或 -1）

        float c_init, c_base, alpha;        ///< PUCT 常数、Dirichlet alpha
        float noise_epsilon;                ///< Dirichlet 噪声混合权重 ε
        float fpu_reduction;                ///< First Play Urgency 衰减系数
        float mlh_slope;                    ///< Moves Left Head 斜率
        float mlh_cap;                      ///< Moves Left Head 最大影响上限
        float mlh_threshold;                ///< MLH Q 阈值：|Q| 低于此值时 M utility 为 0
        float value_decay;                  ///< Backprop 逐层衰减系数：每层将 WDL 向 uniform(⅓) 混合

        /**
         * 构造 MCTS 搜索树。
         * @param c_i       PUCT 初始常数 c_init
         * @param c_b       PUCT 对数基数 c_base
         * @param a         Dirichlet 噪声 alpha（≤0 时禁用噪声）
         * @param noise_eps 噪声混合权重 ε
         * @param fpu_red   FPU 衰减系数
         * @param mlh_slope_ MLH 斜率
         * @param mlh_cap_   MLH 上限
         * @param mlh_threshold_ MLH Q 阈值
         * @param value_decay_ Backprop 逐层衰减系数（1.0=禁用）
         */
        MCTS(float c_i, float c_b, float a, float noise_eps = 0.25f, float fpu_red = 0.4f,
             float mlh_slope_ = 0.0f, float mlh_cap_ = 0.2f, float mlh_threshold_ = 0.8f, float value_decay_ = 1.0f)
            : c_init(c_i), c_base(c_b), alpha(a), noise_epsilon(noise_eps), fpu_reduction(fpu_red),
              mlh_slope(mlh_slope_), mlh_cap(mlh_cap_), mlh_threshold(mlh_threshold_), value_decay(value_decay_)
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

            std::array<int32_t, ACTION_SIZE> valid_children;
            int n_valid = 0;
            for (int a = 0; a < ACTION_SIZE; ++a)
            {
                if (node_pool[root_idx].children[a] != -1)
                    valid_children[n_valid++] = node_pool[root_idx].children[a];
            }
            if (n_valid == 0) return;

            std::array<float, ACTION_SIZE> noise_arr;
            float sum = 0.0f;
            for (int i = 0; i < n_valid; ++i)
            {
                noise_arr[i] = gamma(rng);
                sum += noise_arr[i];
            }
            for (int i = 0; i < n_valid; ++i)
                node_pool[valid_children[i]].noise = noise_arr[i] / (sum + 1e-8f);
        }

        // ======== Selection 辅助函数 ========

        /**
         * 计算 FPU（First Play Urgency）：未访问动作的默认价值估计。
         * 劣势时降低 FPU，鼓励探索新走法。
         *
         * @param node_idx  当前节点索引
         * @param valids    合法动作列表
         * @return FPU 值，用于未访问子节点的 UCB Q 值替代
         */
        float compute_fpu(int32_t node_idx, const ValidMoves<ACTION_SIZE> &valids) const
        {
            const Node &node = node_pool[node_idx];
            float seen_policy = 0.0f;
            for (int action : valids)
            {
                int32_t child_idx = node.children[action];
                if (child_idx != -1 && node_pool[child_idx].n_visits > 0)
                    seen_policy += node_pool[child_idx].prior;
            }
            float scale = (1.0f + node.Q) / 2.0f;
            float effective_fpu = fpu_reduction * scale;
            float fpu_value = node.Q - effective_fpu * std::sqrt(seen_policy);
            return std::max(-1.0f, fpu_value);
        }

        /**
         * 在已展开节点中按 UCB 分数选择最优动作。
         *
         * @param node_idx  当前节点索引
         * @param valids    合法动作列表
         * @param fpu_value FPU 值（未访问子节点的默认 Q 估计）
         * @return 最优动作索引；所有子节点都不存在时返回 -1
         */
        int select_action(int32_t node_idx, const ValidMoves<ACTION_SIZE> &valids, float fpu_value) const
        {
            const Node &node = node_pool[node_idx];
            float parent_n = static_cast<float>(node.n_visits);
            float parent_M = node.M;
            bool is_root = (node_idx == root_idx);

            float best_score = -std::numeric_limits<float>::infinity();
            int best_action = -1;

            for (int action : valids)
            {
                int32_t child_idx = node.children[action];
                if (child_idx != -1)
                {
                    float score = node_pool[child_idx].get_ucb(
                        c_init, c_base, parent_n, is_root, noise_epsilon,
                        fpu_value, parent_M, mlh_slope, mlh_cap, mlh_threshold);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_action = action;
                    }
                }
            }
            return best_action;
        }

        // ======== 核心搜索流程 ========

        /**
         * MCTS 模拟阶段（Selection）：从根节点沿搜索树向下选择，直到到达叶节点或终局状态。
         *
         * 在已展开节点中，按 UCB 分数选择最优动作并前进；
         * 遇到未展开节点或终局状态时停止。
         * 对非终局叶节点可选应用随机对称变换。
         *
         * @param start_state 当前棋盘状态
         * @return SimResult 包含叶节点状态和终局信息
         */
        SimResult<Game> simulate(const Game &start_state)
        {
            sim_env = start_state;
            int32_t curr_idx = root_idx;

            int winner = 0;
            bool board_full = false;

            // Selection：沿已展开路径选择最优动作
            while (node_pool[curr_idx].is_expanded)
            {
                // 快速路径：已标记的终局节点，跳过 check_winner/is_full
                if (node_pool[curr_idx].is_terminal)
                    break;

                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

                float fpu = compute_fpu(curr_idx, valids);
                int action = select_action(curr_idx, valids, fpu);
                if (action == -1) break;

                sim_env.step(action);
                curr_idx = node_pool[curr_idx].children[action];

                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
                if (winner != 0 || board_full)
                {
                    // 标记终局节点
                    WDLValue tw = (winner != 0) ? winner_to_wdl(winner) : WDLValue::draw();
                    node_pool[curr_idx].is_terminal = true;
                    node_pool[curr_idx].terminal_wdl = tw;
                    break;
                }
            }
            current_leaf_idx = curr_idx;
            current_leaf_turn = sim_env.get_turn();

            // 快速路径：已标记的终局节点
            if (node_pool[curr_idx].is_terminal)
                return {sim_env, node_pool[curr_idx].terminal_wdl, true};

            // 首次终局检测（循环未进入或因非终局条件退出）
            if (winner == 0 && !board_full)
            {
                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
            }

            if (winner != 0)
            {
                WDLValue tw = winner_to_wdl(winner);
                node_pool[curr_idx].is_terminal = true;
                node_pool[curr_idx].terminal_wdl = tw;
                return {sim_env, tw, true};
            }
            if (board_full)
            {
                node_pool[curr_idx].is_terminal = true;
                node_pool[curr_idx].terminal_wdl = WDLValue::draw();
                return {sim_env, WDLValue::draw(), true};
            }

            // 非终局叶节点：返回未变换的状态（对称变换由 BatchedMCTS 层处理）
            return {sim_env, WDLValue{}, false};
        }

        // ======== Backpropagation 辅助函数 ========

        /**
         * 展开叶节点：为所有合法动作创建子节点，设置策略先验和可选 Dirichlet 噪声。
         *
         * 流程：
         *   1. 对 policy 应用对称逆变换（恢复 simulate 中的随机对称）
         *   2. 归一化策略概率
         *   3. 根节点首次展开时生成 Dirichlet 噪声
         *   4. 为每个合法动作分配子节点
         *
         * @param policy_logits NN 策略输出（或 uniform prior），长度 ACTION_SIZE
         */
        void expand_leaf(std::span<const float> policy_logits)
        {
            // policy 拷贝（对称逆变换已上提到 BatchedMCTS 层）
            std::array<float, ACTION_SIZE> policy;
            std::copy(policy_logits.begin(), policy_logits.end(), policy.begin());

            auto valids = sim_env.get_valid_moves();

            // 策略归一化：只对合法动作的 logits 求和
            float policy_sum = 0.0f;
            for (int action : valids)
                policy_sum += policy[action];

            // 根节点首次展开时生成 Dirichlet 噪声
            std::array<float, ACTION_SIZE> noise_arr{};
            bool has_noise = (node_pool[current_leaf_idx].parent == -1 && alpha > 0.0f);
            if (has_noise)
            {
                std::mt19937 &rng = get_rng();
                std::gamma_distribution<float> gamma(alpha, 1.0f);
                float sum = 0.0f;
                for (int i = 0; i < valids.size(); ++i)
                {
                    noise_arr[i] = gamma(rng);
                    sum += noise_arr[i];
                }
                for (int i = 0; i < valids.size(); ++i)
                    noise_arr[i] /= (sum + 1e-8f);
            }

            // 展开：为每个合法动作创建子节点
            int noise_idx = 0;
            for (int action : valids)
            {
                float prob = policy[action] / (policy_sum + 1e-8f);
                if (node_pool[current_leaf_idx].children[action] == -1)
                {
                    int32_t new_node = allocate_node(current_leaf_idx, prob);
                    node_pool[current_leaf_idx].children[action] = new_node;
                    if (has_noise)
                        node_pool[new_node].noise = noise_arr[noise_idx];
                }
                ++noise_idx;
            }
            node_pool[current_leaf_idx].is_expanded = true;
        }

        /**
         * 从叶节点沿 parent 链向上更新统计量（绝对视角 WDL 不交换）。
         * Q 在每层根据该节点的落子方计算。
         * moves_left 递增 1（父节点比子节点多一步）。
         *
         * @param leaf_wdl   叶节点评估的 WDL（绝对视角）
         * @param moves_left 预期剩余步数
         */
        void propagate(WDLValue leaf_wdl, float moves_left)
        {
            int32_t idx = current_leaf_idx;
            float ml = moves_left;
            int turn = current_leaf_turn;
            while (idx != -1)
            {
                Node &node = node_pool[idx];
                node.n_visits++;
                node.wdl.update_mean(leaf_wdl, node.n_visits);
                node.Q = node.wdl.q(turn);
                node.M += (ml - node.M) / node.n_visits;
                ml += 1.0f;
                turn = -turn;
                idx = node.parent;

                if (value_decay < 1.0f)
                    leaf_wdl = leaf_wdl.decayed(value_decay);
            }
        }

        // ======== 公共反向传播接口 ========

        /**
         * 反向传播阶段：用评估结果展开叶节点并更新路径上的统计量。
         *
         * @param policy_logits NN 输出的策略 logits（或 uniform），长度 ACTION_SIZE
         * @param wdl           WDL 评估值（绝对视角）
         * @param moves_left    预期剩余步数（NN 输出，纯 MCTS 时为 0）
         * @param is_terminal   是否为终局状态（终局不展开，仅传播）
         */
        void backprop(std::span<const float> policy_logits, WDLValue wdl, float moves_left, bool is_terminal)
        {
            if (current_leaf_idx == -1) return;

            if (!is_terminal)
                expand_leaf(policy_logits);

            propagate(wdl, is_terminal ? 0.0f : moves_left);
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
         * 布局：[root_N, root_Q, root_M, root_D, root_P1W, root_P2W,
         *         a0_N, a0_Q, a0_prior, a0_noise, a0_M, a0_D, a0_P1W, a0_P2W, a1_N, ...]
         * 总长度 = 6 + ACTION_SIZE × 8
         *
         * @param out 预分配的输出缓冲区
         */
        void get_root_stats(float *out) const
        {
            const Node &root = node_pool[root_idx];
            out[0] = static_cast<float>(root.n_visits);
            out[1] = root.Q;
            out[2] = root.M;
            out[3] = root.wdl.d;
            out[4] = root.wdl.p1w;
            out[5] = root.wdl.p2w;

            float *p = out + 6;
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
                    p[5] = child.wdl.d;
                    p[6] = child.wdl.p1w;
                    p[7] = child.wdl.p2w;
                }
                else
                {
                    std::fill(p, p + 8, 0.0f);
                }
                p += 8;
            }
        }
    };
}
