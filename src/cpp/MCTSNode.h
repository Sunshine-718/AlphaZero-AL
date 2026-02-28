#ifndef F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#define F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace AlphaZero
{
    /**
     * MCTS 搜索树节点。
     *
     * 存储于 MCTS::node_pool（线性数组），通过 int32_t 索引相互引用。
     * 每个节点对应搜索树中的一个游戏状态，记录访问次数、Q 值、策略先验等统计量。
     *
     * @tparam ACTION_SIZE 动作空间大小（如 Connect4 = 7）
     */
    template <int ACTION_SIZE>
    struct MCTSNode
    {
        int32_t parent = -1;                        ///< 父节点索引，根节点为 -1
        std::array<int32_t, ACTION_SIZE> children;  ///< 子节点索引数组，-1 表示未创建

        int n_visits = 0;       ///< 访问次数 N
        float Q = 0.0f;         ///< 状态价值 (当前落子方视角，由 propagate 计算)
        float d = 0.0f;         ///< 和棋率运行均值（绝对视角）
        float p1w = 0.0f;       ///< P1 胜率运行均值（绝对视角）
        float p2w = 0.0f;       ///< P2 胜率运行均值（绝对视角）
        float M = 0.0f;         ///< 预期剩余步数的运行均值（Moves Left Head）
        float prior = 0.0f;     ///< 神经网络策略先验概率 P(a)
        float noise = 0.0f;     ///< Dirichlet 噪声（仅根节点的子节点有效）
        bool is_expanded = false;   ///< 是否已展开（子节点已创建）

        MCTSNode() {
            children.fill(-1);
        }

        /**
         * 重置节点状态，以便在节点池中复用。
         * @param p      父节点索引
         * @param p_prior 策略先验概率
         */
        void reset(int32_t p, float p_prior) {
            parent = p;
            prior = p_prior;
            children.fill(-1);
            n_visits = 0;
            Q = 0.0f;
            d = 0.0f;
            p1w = 0.0f;
            p2w = 0.0f;
            M = 0.0f;
            noise = 0.0f;
            is_expanded = false;
        }

        /**
         * 计算 UCB 分数，用于在树遍历中选择最优动作。
         *
         * UCB = q_value + u_score + m_utility
         *   q_value   = -Q（父节点视角：子节点越差，父节点越好）
         *   u_score   = C_puct × prior × √N_parent / (1 + N_child)（PUCT 探索项）
         *   m_utility = clamp(slope × (child_M - parent_M), -cap, cap) × Q × ramp(|Q|, threshold)
         *
         * @param c_init        PUCT 初始常数
         * @param c_base        PUCT 对数基数
         * @param parent_n      父节点访问次数
         * @param is_root_node  是否为根节点（根节点混合 Dirichlet 噪声）
         * @param noise_epsilon 噪声混合权重 ε
         * @param fpu_value     First Play Urgency 值（未访问节点的默认 Q 估计）
         * @param parent_M      父节点的 M 值
         * @param mlh_slope     MLH 斜率（0 = 禁用 MLH）
         * @param mlh_cap       MLH 最大影响上限
         * @param mlh_threshold MLH Q 阈值：|Q| 低于此值时 M utility 为 0（0 = 无阈值）
         * @return UCB 分数
         */
        [[nodiscard]] float get_ucb(float c_init, float c_base, float parent_n,
                                     bool is_root_node, float noise_epsilon, float fpu_value,
                                     float parent_M, float mlh_slope, float mlh_cap,
                                     float mlh_threshold) const {
            // 根节点混合 Dirichlet 噪声：effective_prior = (1-ε)·prior + ε·noise
            float effective_prior = prior;
            if (is_root_node) {
                effective_prior = (1.0f - noise_epsilon) * prior + noise_epsilon * noise;
            }

            // Q 值：未访问节点用 FPU 替代，已访问节点取 -Q（翻转到父节点视角）
            float q_value = (n_visits == 0) ? fpu_value : -Q;
            // PUCT 探索项：C(s) = c_init + log((N_parent + c_base + 1) / c_base)
            float c_puct = c_init + std::log((parent_n + c_base + 1.0f) / c_base);
            float u_score = c_puct * effective_prior * std::sqrt(parent_n) / (1.0f + n_visits);

            // MLH 偏好项：鼓励在劣势时快速结束、优势时延长博弈
            //   Q < 0 (对手赢着): 快结束 bonus, 慢结束 penalty
            //   Q > 0 (对手输着): 慢结束 bonus, 快结束 penalty
            //   Q ≈ 0 (均势):    自然无效果
            // mlh_threshold: |Q| 低于阈值时完全抑制 M utility，高于阈值后平滑 ramp-up
            float m_utility = 0.0f;
            if (mlh_slope > 0.0f && n_visits > 0) {
                float abs_q = std::abs(Q);
                if (mlh_threshold <= 0.0f || abs_q >= mlh_threshold) {
                    float m_diff = M - parent_M;
                    m_utility = std::clamp(mlh_slope * m_diff, -mlh_cap, mlh_cap) * Q;
                    // 平滑 ramp-up：|Q| 从 threshold 到 1.0 线性过渡
                    if (mlh_threshold > 0.0f && mlh_threshold < 1.0f) {
                        m_utility *= (abs_q - mlh_threshold) / (1.0f - mlh_threshold);
                    }
                }
            }

            return q_value + u_score + m_utility;
        }
    };
}
#endif
