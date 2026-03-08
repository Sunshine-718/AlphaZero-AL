#pragma once
#include "GameContext.h"
#include "MCTSNode.h"
#include <algorithm>
#include <limits>
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
     */
    template <MCTSGame Game>
    struct SimResult
    {
        Game board;
        WDLValue terminal_wdl;
        bool is_terminal;
    };

    /// Virtual Loss 路径条目：记录经过的 (parent_node, edge_idx) 对
    struct VLPathEntry
    {
        int32_t node_idx;   ///< 父节点索引
        int32_t edge_idx;   ///< 在该节点中选择的边索引
    };

    /**
     * 单棵 MCTS 搜索树，对应一局游戏。
     *
     * 核心流程：simulate() 选择叶节点 → 外部评估 → backprop() 反向传播。
     * 节点和边分离存储在 NodePool 中，通过 int32_t 索引引用。
     * 子节点延迟分配：expand_leaf() 只创建 Edge，子 Node 在首次访问时分配。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class MCTS
    {
    public:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;

        NodePool pool_;                     ///< 节点 + 边池
        int32_t root_idx_ = -1;             ///< 根节点索引
        const SearchConfig *config_;        ///< 搜索配置（非拥有指针）

        Game sim_env;                       ///< simulate() 中用于模拟走步的临时游戏状态
        int32_t current_leaf_idx = -1;      ///< 最近一次 simulate() 到达的叶节点索引
        int current_leaf_turn = 1;          ///< 最近一次 simulate() 叶节点的落子方

        // ======== Virtual Loss 状态 ========
        std::vector<std::vector<VLPathEntry>> vl_paths_;   ///< K 条路径，每条记录 (parent_node, edge_idx)
        std::vector<Game>     vl_envs_;                     ///< K 个叶节点的游戏状态
        std::vector<int32_t>  vl_leaf_indices_;             ///< K 个叶节点索引
        std::vector<int>      vl_leaf_turns_;               ///< K 个叶节点的落子方

        /**
         * 构造 MCTS 搜索树。
         * @param config 搜索配置的引用（由 BatchedMCTS 拥有）
         */
        MCTS(const SearchConfig &config)
            : config_(&config), pool_(2048)
        {
            reset();
        }

        /// 重置搜索树：清空池，创建新根节点
        void reset()
        {
            pool_.reset();
            root_idx_ = pool_.allocate_node();
            pool_.node(root_idx_).turn = 1; // 默认 P1 先手
        }

        // ======== 树剪枝 ========

        /**
         * 落子后将对应子树提升为新根节点。
         * 遍历根节点的 edges 查找匹配动作（O(num_edges)，每步仅调用一次）。
         */
        void prune_root(int action)
        {
            MCTSNode &root = pool_.node(root_idx_);
            if (root.is_expanded)
            {
                for (int i = 0; i < root.num_edges; ++i)
                {
                    Edge &e = pool_.edge(root.edge_offset, i);
                    if (e.action == action && e.child != -1)
                    {
                        root_idx_ = e.child;
                        pool_.node(root_idx_).parent = -1;
                        apply_root_noise();
                        return;
                    }
                }
            }
            reset();
        }

        /**
         * 为根节点的边重新生成 Dirichlet 噪声。
         */
        void apply_root_noise()
        {
            if (config_->dirichlet_alpha <= 0.0f) return;
            MCTSNode &root = pool_.node(root_idx_);
            if (!root.is_expanded || root.num_edges == 0) return;

            std::mt19937 &rng = get_rng();
            std::gamma_distribution<float> gamma(config_->dirichlet_alpha, 1.0f);

            float sum = 0.0f;
            std::vector<float> noise_arr(root.num_edges);
            for (int i = 0; i < root.num_edges; ++i)
            {
                noise_arr[i] = gamma(rng);
                sum += noise_arr[i];
            }
            float inv = 1.0f / (sum + 1e-8f);
            for (int i = 0; i < root.num_edges; ++i)
                pool_.edge(root.edge_offset, i).noise = noise_arr[i] * inv;
        }

        // ======== Selection 辅助函数 ========

        /**
         * 计算 FPU（First Play Urgency）：未访问动作的默认价值估计。
         * 遍历节点的 edges，统计已访问子节点的 policy 总和。
         */
        float compute_fpu(int32_t node_idx) const
        {
            const MCTSNode &node = pool_.node(node_idx);
            float parent_q = node.mean_q(); // 只用真实访问统计
            float seen_policy = 0.0f;
            for (int i = 0; i < node.num_edges; ++i)
            {
                const Edge &e = pool_.edge(node.edge_offset, i);
                // 只统计有真实回传的子节点，in-flight 不算 seen
                if (e.child != -1 && pool_.node(e.child).n_visits > 0)
                    seen_policy += e.prior;
            }
            float scale = (1.0f + parent_q) / 2.0f;
            float effective_fpu = config_->fpu_reduction * scale;
            float fpu_value = parent_q - effective_fpu * std::sqrt(seen_policy);
            return std::max(-1.0f, fpu_value);
        }

        /**
         * 在已展开节点中选择最优边（UCB 分数最高）。
         * UCB = q_value + u_score + m_utility，内联计算。
         * @return 最优边索引；无有效边时返回 -1
         */
        int select_edge(int32_t node_idx, float fpu_value) const
        {
            const MCTSNode &node = pool_.node(node_idx);
            // parent_n = real + inflight：VL 期间正确反映"已派出"的模拟数
            float parent_n = static_cast<float>(node.n_visits + node.n_inflight);
            float parent_M = node.mean_M();
            bool is_root = (node_idx == root_idx_);
            float ne = config_->noise_epsilon;

            float best_score = -std::numeric_limits<float>::infinity();
            int best_edge = -1;

            for (int i = 0; i < node.num_edges; ++i)
            {
                const Edge &e = pool_.edge(node.edge_offset, i);

                // 有效先验：根节点混合 Dirichlet 噪声
                float effective_prior = e.prior;
                if (is_root && ne > 0.0f)
                    effective_prior = (1.0f - ne) * e.prior + ne * e.noise;

                // Q 值：只看真实回传；in-flight 节点用 FPU
                float q_value;
                float child_Q = 0.0f;
                float child_M = 0.0f;
                int child_visits_total = 0; // real + inflight，用于探索项分母

                if (e.child != -1 && pool_.node(e.child).n_visits > 0)
                {
                    // 有真实回传：Q/M 用真实统计，探索项分母用 real+inflight
                    const MCTSNode &child = pool_.node(e.child);
                    child_visits_total = child.n_visits + child.n_inflight;
                    child_Q = child.mean_q(); // 只用真实 W/N，子节点视角
                    child_M = child.mean_M();
                    q_value = -child_Q; // 翻转到父节点视角
                }
                else if (e.child != -1 && pool_.node(e.child).n_inflight > 0)
                {
                    // 只有 in-flight、没有真实回传：Q 用 FPU，探索分母用 inflight
                    q_value = fpu_value;
                    child_visits_total = pool_.node(e.child).n_inflight;
                }
                else
                {
                    q_value = fpu_value;
                }

                // PUCT 探索项
                float c_puct = config_->c_init +
                    std::log((parent_n + config_->c_base + 1.0f) / config_->c_base);
                float u_score = c_puct * effective_prior *
                    std::sqrt(parent_n) / (1.0f + child_visits_total);

                // MLH 偏好项（只对有真实访问的子节点生效）
                float m_utility = 0.0f;
                if (config_->mlh_slope > 0.0f && e.child != -1 && pool_.node(e.child).n_visits > 0)
                {
                    float abs_q = std::abs(child_Q);
                    if (config_->mlh_threshold <= 0.0f || abs_q >= config_->mlh_threshold)
                    {
                        float m_diff = child_M - parent_M;
                        m_utility = std::clamp(config_->mlh_slope * m_diff,
                                               -config_->mlh_cap, config_->mlh_cap) * child_Q;
                        if (config_->mlh_threshold > 0.0f && config_->mlh_threshold < 1.0f)
                            m_utility *= (abs_q - config_->mlh_threshold) /
                                         (1.0f - config_->mlh_threshold);
                    }
                }

                float score = q_value + u_score + m_utility;
                if (score > best_score)
                {
                    best_score = score;
                    best_edge = i;
                }
            }
            return best_edge;
        }

        // ======== 核心搜索流程 ========

        /**
         * MCTS Selection：从根节点沿搜索树向下选择，直到叶节点或终局。
         * 子节点延迟分配：首次沿某条边访问时才创建子 Node。
         */
        SimResult<Game> simulate(const Game &start_state)
        {
            sim_env = start_state;
            int32_t curr_idx = root_idx_;

            int winner = 0;
            bool board_full = false;

            while (pool_.node(curr_idx).is_expanded)
            {
                MCTSNode &node = pool_.node(curr_idx);

                // 快速路径：已标记的终局节点
                if (node.is_terminal)
                    break;

                if (node.num_edges == 0) break;

                float fpu = compute_fpu(curr_idx);
                int best_edge = select_edge(curr_idx, fpu);
                if (best_edge < 0) break;

                Edge &e = pool_.edge(node.edge_offset, best_edge);
                sim_env.step(e.action);

                // 延迟子节点分配
                if (e.child == -1)
                {
                    e.child = pool_.allocate_node();
                    MCTSNode &child = pool_.node(e.child);
                    child.parent = curr_idx;
                    child.parent_edge_idx = best_edge;
                    child.turn = sim_env.get_turn();
                }
                curr_idx = e.child;

                // 终局检测
                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
                if (winner != 0 || board_full)
                {
                    WDLValue tw = (winner != 0) ? winner_to_wdl(winner) : WDLValue::draw();
                    MCTSNode &leaf = pool_.node(curr_idx);
                    leaf.is_terminal = true;
                    leaf.set_terminal_wdl(tw);
                    break;
                }
            }

            current_leaf_idx = curr_idx;
            current_leaf_turn = sim_env.get_turn();

            // 快速路径：已标记的终局节点
            if (pool_.node(curr_idx).is_terminal)
                return {sim_env, pool_.node(curr_idx).get_terminal_wdl(), true};

            // 首次终局检测
            if (winner == 0 && !board_full)
            {
                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
            }

            if (winner != 0)
            {
                WDLValue tw = winner_to_wdl(winner);
                MCTSNode &leaf = pool_.node(curr_idx);
                leaf.is_terminal = true;
                leaf.set_terminal_wdl(tw);
                return {sim_env, tw, true};
            }
            if (board_full)
            {
                MCTSNode &leaf = pool_.node(curr_idx);
                leaf.is_terminal = true;
                leaf.set_terminal_wdl(WDLValue::draw());
                return {sim_env, WDLValue::draw(), true};
            }

            return {sim_env, WDLValue{}, false};
        }

        // ======== Expansion & Backpropagation ========

        /**
         * 展开叶节点：为所有合法动作创建 Edge（不创建子 Node）。
         */
        void expand_leaf(std::span<const float> policy_logits)
        {
            std::array<float, ACTION_SIZE> policy;
            std::copy(policy_logits.begin(), policy_logits.end(), policy.begin());

            auto valids = sim_env.get_valid_moves();
            int num_valid = valids.size();

            MCTSNode &leaf = pool_.node(current_leaf_idx);
            leaf.edge_offset = pool_.allocate_edges(num_valid);
            leaf.num_edges = num_valid;
            leaf.is_expanded = true;

            // 策略归一化
            float policy_sum = 0.0f;
            for (int action : valids)
                policy_sum += policy[action];

            // 根节点首次展开时生成 Dirichlet 噪声
            std::array<float, ACTION_SIZE> noise_arr{};
            bool has_noise = (leaf.parent == -1 && config_->dirichlet_alpha > 0.0f);
            if (has_noise)
            {
                std::mt19937 &rng = get_rng();
                std::gamma_distribution<float> gamma(config_->dirichlet_alpha, 1.0f);
                float sum = 0.0f;
                for (int i = 0; i < num_valid; ++i)
                {
                    noise_arr[i] = gamma(rng);
                    sum += noise_arr[i];
                }
                float inv = 1.0f / (sum + 1e-8f);
                for (int i = 0; i < num_valid; ++i)
                    noise_arr[i] *= inv;
            }

            // 填充 Edge
            for (int i = 0; i < num_valid; ++i)
            {
                Edge &e = pool_.edge(leaf.edge_offset, i);
                e.action = valids.moves[i];
                e.prior = policy[valids.moves[i]] / (policy_sum + 1e-8f);
                e.child = -1; // 延迟分配
                if (has_noise)
                    e.noise = noise_arr[i];
            }
        }

        /**
         * 从叶节点沿 parent 链向上累加 WDL 和 M。
         * WDL 使用绝对视角累加，不翻转。Q 按需从 mean_wdl() 计算。
         */
        void propagate(WDLValue leaf_wdl, float moves_left)
        {
            int32_t idx = current_leaf_idx;
            float ml = moves_left;
            while (idx != -1)
            {
                MCTSNode &node = pool_.node(idx);
                node.n_visits++;
                node.W_d   += leaf_wdl.d;
                node.W_p1w += leaf_wdl.p1w;
                node.W_p2w += leaf_wdl.p2w;
                node.M_sum += ml;
                ml += 1.0f;
                idx = node.parent;

                if (config_->value_decay < 1.0f)
                    leaf_wdl = leaf_wdl.decayed(config_->value_decay);
            }
        }

        /**
         * 反向传播：展开叶节点并更新路径统计量。
         */
        void backprop(std::span<const float> policy_logits, WDLValue wdl, float moves_left, bool is_terminal)
        {
            if (current_leaf_idx == -1) return;
            if (!is_terminal)
                expand_leaf(policy_logits);
            propagate(wdl, is_terminal ? 0.0f : moves_left);
        }

        // ======== Virtual Loss 方法 ========

        /**
         * 预分配 VL 状态向量。在每轮 VL 搜索开始前调用。
         * @param K 每棵树每次迭代的 VL 模拟次数
         */
        void prepare_vl(int K)
        {
            vl_paths_.resize(K);
            vl_envs_.resize(K);
            vl_leaf_indices_.resize(K, -1);
            vl_leaf_turns_.resize(K, 1);
            for (int k = 0; k < K; ++k)
                vl_paths_[k].clear();
        }

        /**
         * VL 模拟：与 simulate() 逻辑相同，额外施加 Virtual Loss (in-flight)。
         *
         * 每个节点 n_inflight 只加一次——到达时（作为 child）加。
         * root 没有 parent，在循环前单独加。
         * 这样路径 root→A→B 的 inflight 分布为 root=1, A=1, B=1。
         *
         * Q 值只看真实回传 (n_visits)，不受 inflight 影响。
         *
         * @param k 第 k 次 VL 模拟（0-indexed）
         * @param start_state 当前棋盘状态
         */
        SimResult<Game> simulate_vl(int k, const Game &start_state)
        {
            vl_paths_[k].clear();
            sim_env = start_state;
            int32_t curr_idx = root_idx_;

            int winner = 0;
            bool board_full = false;

            // Root VL：root 没有 parent，在循环前单独施加。
            // 用途：select_edge 的 parent_n = n_visits + n_inflight 正确反映
            // 从 root 派出的 in-flight 模拟数。
            // 若循环未执行（root 未展开/终局），vl_paths_ 为空，
            // remove_all_vl 不会尝试移除 root VL（由 empty 检测保护）。
            bool root_vl_applied = false;

            while (pool_.node(curr_idx).is_expanded)
            {
                MCTSNode &node = pool_.node(curr_idx);

                if (node.is_terminal) break;
                if (node.num_edges == 0) break;

                float fpu = compute_fpu(curr_idx);
                int best_edge = select_edge(curr_idx, fpu);
                if (best_edge < 0) break;

                // 首次进入循环时给 root 施加 VL
                if (!root_vl_applied)
                {
                    pool_.node(root_idx_).n_inflight += config_->vl_count;
                    root_vl_applied = true;
                }

                Edge &e = pool_.edge(node.edge_offset, best_edge);
                sim_env.step(e.action);

                // 延迟子节点分配
                if (e.child == -1)
                {
                    e.child = pool_.allocate_node();
                    MCTSNode &child = pool_.node(e.child);
                    child.parent = curr_idx;
                    child.parent_edge_idx = best_edge;
                    child.turn = sim_env.get_turn();
                }

                // 施加 Virtual Loss：只给 child（到达时加）
                // child 的 n_inflight 在后续循环作为 parent_n 和 child_visits_total 使用
                pool_.node(e.child).n_inflight += config_->vl_count;

                // 记录路径
                vl_paths_[k].push_back({curr_idx, best_edge});

                curr_idx = e.child;

                // 终局检测
                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
                if (winner != 0 || board_full)
                {
                    WDLValue tw = (winner != 0) ? winner_to_wdl(winner) : WDLValue::draw();
                    MCTSNode &leaf = pool_.node(curr_idx);
                    leaf.is_terminal = true;
                    leaf.set_terminal_wdl(tw);
                    break;
                }
            }

            // 存储叶节点状态
            vl_leaf_indices_[k] = curr_idx;
            vl_leaf_turns_[k] = sim_env.get_turn();
            vl_envs_[k] = sim_env;

            // 快速路径：已标记的终局节点
            if (pool_.node(curr_idx).is_terminal)
                return {sim_env, pool_.node(curr_idx).get_terminal_wdl(), true};

            // 首次终局检测
            if (winner == 0 && !board_full)
            {
                winner = sim_env.check_winner();
                board_full = sim_env.is_full();
            }

            if (winner != 0)
            {
                WDLValue tw = winner_to_wdl(winner);
                MCTSNode &leaf = pool_.node(curr_idx);
                leaf.is_terminal = true;
                leaf.set_terminal_wdl(tw);
                return {sim_env, tw, true};
            }
            if (board_full)
            {
                MCTSNode &leaf = pool_.node(curr_idx);
                leaf.is_terminal = true;
                leaf.set_terminal_wdl(WDLValue::draw());
                return {sim_env, WDLValue::draw(), true};
            }

            return {sim_env, WDLValue{}, false};
        }

        /**
         * 移除所有 VL：恢复 K 条路径上的节点 n_inflight。
         * 必须在 backprop_vl 之前调用，确保树统计量干净。
         *
         * 与 simulate_vl 对称：
         *  - 非空路径 → root VL 存在 → 移除
         *  - 每条边只有 child 被加了 VL → 只移除 child
         *
         * 安全性：
         *  - K 钳位到 vl_paths_ 实际大小，防止越界
         *  - 移除后清空路径，使重复调用变成幂等 no-op
         *
         * @param K VL 模拟次数
         */
        void remove_all_vl(int K)
        {
            int safe_K = std::min(K, static_cast<int>(vl_paths_.size()));
            int vl = config_->vl_count;
            for (int k = 0; k < safe_K; ++k)
            {
                // 路径非空 → 该次模拟进入了选择循环 → root 被施加了 VL
                if (!vl_paths_[k].empty())
                    pool_.node(root_idx_).n_inflight -= vl;

                for (auto &entry : vl_paths_[k])
                {
                    // 只移除 child VL（simulate_vl 只给 child 加了 VL）
                    Edge &e = pool_.edge(
                        pool_.node(entry.node_idx).edge_offset, entry.edge_idx);
                    if (e.child != -1)
                        pool_.node(e.child).n_inflight -= vl;
                }
                vl_paths_[k].clear();  // 幂等：重复调用不会重复减
            }
        }

        /**
         * VL 反向传播：恢复第 k 次模拟的叶节点状态，执行展开 + 传播。
         *
         * 处理重复叶节点：若两次 VL 模拟到达同一未展开叶节点，
         * 第一次 backprop_vl 展开它，第二次检测到 is_expanded=true 后跳过展开。
         *
         * @param k 第 k 次 VL 模拟（0-indexed）
         */
        void backprop_vl(int k, std::span<const float> policy_logits,
                         WDLValue wdl, float moves_left, bool is_terminal)
        {
            // 恢复第 k 次模拟的叶节点状态
            current_leaf_idx = vl_leaf_indices_[k];
            current_leaf_turn = vl_leaf_turns_[k];
            sim_env = vl_envs_[k];

            if (current_leaf_idx == -1) return;

            if (!is_terminal)
            {
                MCTSNode &leaf = pool_.node(current_leaf_idx);
                if (!leaf.is_expanded)
                    expand_leaf(policy_logits);
                // 若已被之前的 backprop_vl 展开，跳过展开
            }
            propagate(wdl, is_terminal ? 0.0f : moves_left);
        }

        // ======== 统计查询 ========

        /**
         * 根节点各动作的访问次数。
         * 遍历 edges，按 action 索引填充 counts。
         */
        std::vector<int> get_counts() const
        {
            std::vector<int> counts(ACTION_SIZE, 0);
            const MCTSNode &root = pool_.node(root_idx_);
            if (!root.is_expanded) return counts;

            for (int i = 0; i < root.num_edges; ++i)
            {
                const Edge &e = pool_.edge(root.edge_offset, i);
                if (e.child != -1)
                    counts[e.action] = pool_.node(e.child).n_visits;
            }
            return counts;
        }

        /**
         * 根节点统计量写入 flat 缓冲区。
         * 布局：[root_N, root_Q, root_M, root_D, root_P1W, root_P2W,
         *         a0_N, a0_Q, a0_prior, a0_noise, a0_M, a0_D, a0_P1W, a0_P2W, ...]
         */
        void get_root_stats(float *out) const
        {
            const MCTSNode &root = pool_.node(root_idx_);
            WDLValue root_wdl = root.mean_wdl();
            out[0] = static_cast<float>(root.n_visits);
            out[1] = root.mean_q();
            out[2] = root.mean_M();
            out[3] = root_wdl.d;
            out[4] = root_wdl.p1w;
            out[5] = root_wdl.p2w;

            float *p = out + 6;
            std::fill(p, p + ACTION_SIZE * 8, 0.0f);

            if (!root.is_expanded) return;
            for (int i = 0; i < root.num_edges; ++i)
            {
                const Edge &e = pool_.edge(root.edge_offset, i);
                float *slot = p + e.action * 8;
                slot[2] = e.prior;
                slot[3] = e.noise;
                if (e.child != -1)
                {
                    const MCTSNode &child = pool_.node(e.child);
                    WDLValue cw = child.mean_wdl();
                    slot[0] = static_cast<float>(child.n_visits);
                    slot[1] = child.mean_q();
                    slot[4] = child.mean_M();
                    slot[5] = cw.d;
                    slot[6] = cw.p1w;
                    slot[7] = cw.p2w;
                }
            }
        }
    };
}
