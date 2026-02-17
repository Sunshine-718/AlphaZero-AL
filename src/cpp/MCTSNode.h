#ifndef F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#define F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <cstdint>

namespace AlphaZero
{
    template <int ACTION_SIZE>
    struct MCTSNode
    {
        int32_t parent = -1;
        // 存储子节点在数组中的索引，初始化为 -1
        std::array<int32_t, ACTION_SIZE> children;

        int n_visits = 0;
        float Q = 0.0f;
        float prior = 0.0f;
        float noise = 0.0f;
        bool is_expanded = false;

        MCTSNode() {
            children.fill(-1);
        }

        // 重置节点状态，以便在节点池中复用
        void reset(int32_t p, float p_prior) {
            parent = p;
            prior = p_prior;
            children.fill(-1);
            n_visits = 0;
            Q = 0.0f;
            noise = 0.0f;
            is_expanded = false;
        }

        // 计算 UCB 时需要传入父节点的访问次数，因为现在不通过指针找 parent
        [[nodiscard]] float get_ucb(float c_init, float c_base, float parent_n,
                                     bool is_root_node, float noise_epsilon) const {
            float effective_prior = prior;
            if (is_root_node) {
                effective_prior = (1.0f - noise_epsilon) * prior + noise_epsilon * noise;
            }

            float c_puct = c_init + std::log((parent_n + c_base + 1.0f) / c_base);
            float u_score = c_puct * effective_prior * std::sqrt(parent_n) / (1.0f + n_visits);
            return -Q + u_score;
        }
    };
}
#endif
