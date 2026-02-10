#ifndef F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#define F7A41D12_1F96_492F_AB14_DA76ADFB91E6
#pragma once
#include "Constants.h"
#include <array>
#include <cmath>
#include <limits>
#include <memory>

namespace AlphaZero
{
    struct MCTSNode
    {
        MCTSNode *parent = nullptr;
        std::array<std::unique_ptr<MCTSNode>, Config::ACTION_SIZE> children;

        int n_visits = 0;
        float Q = 0.0f;
        float prior = 0.0f;

        float noise = 0.0f;

        bool is_expanded = false;

        MCTSNode(MCTSNode *p, float prior_prob)
            : parent(p), prior(prior_prob) {}
        
        [[nodiscard]] bool is_root() const {
            return parent == nullptr;
        }
        
        [[nodiscard]] float get_ucb(float c_init, float c_base, float parent_n) const {
            float effective_prior = prior;
            if (parent && parent->is_root()) {
                effective_prior = (1.0f - Config::NOISE_EPSILON) * prior + Config::NOISE_EPSILON * noise;
            }
            float u_score;
            if (n_visits == 0) {
                u_score = std::numeric_limits<float>::infinity();
            } else {
                float c_puct = c_init + std::log((parent_n + c_base + 1.0f) / c_base);
                u_score = c_puct * effective_prior * std::sqrt(parent_n) / (1.0f + n_visits);
            }
            return -Q + u_score;
        }

        void update(float leaf_value, float discount) {
            if (parent) {
                parent->update(-leaf_value * discount, discount);
            }
            n_visits++;
            Q += (leaf_value - Q) / n_visits;
        }
    };
}

#endif /* F7A41D12_1F96_492F_AB14_DA76ADFB91E6 */
