#ifndef D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#define D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#pragma once
#include "Connect4.h"
#include "MCTSNode.h"
#include <optional>
#include <random>
#include <vector>

namespace AlphaZero
{
    inline std::mt19937 &get_rng()
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        return rng;
    }

    class MCTS
    {
    public:
        // 节点池
        std::vector<MCTSNode> node_pool;
        int32_t root_idx = 0;
        int32_t next_free_node = 0;

        Connect4 sim_env;
        int32_t current_leaf_idx = -1;
        bool current_flipped = false;

        float c_init, c_base, discount, alpha;

        MCTS(float c_i, float c_b, float disc, float a)
            : c_init(c_i), c_base(c_b), discount(disc), alpha(a)
        {
            // 预分配内存，减少 search 过程中的扩容
            node_pool.resize(2000); 
            reset();
        }

        void reset()
        {
            next_free_node = 0;
            root_idx = allocate_node(-1, 1.0f);
        }

        int32_t allocate_node(int32_t parent, float prior) {
            if (next_free_node >= node_pool.size()) {
                node_pool.resize(node_pool.size() * 2);
            }
            int32_t idx = next_free_node++;
            node_pool[idx].reset(parent, prior);
            return idx;
        }

        void prune_root(int action)
        {
            int32_t child_idx = node_pool[root_idx].children[action];
            if (action >= 0 && action < Config::ACTION_SIZE && child_idx != -1)
            {
                root_idx = child_idx;
                node_pool[root_idx].parent = -1;
                // 注意：数组结构下，剪枝后的老节点依然占据空间，直到下次 reset()
            }
            else
            {
                reset();
            }
        }

        void simulate(const Connect4 &start_state, Connect4 &out_nn_input_board, bool &out_is_terminal, float &out_terminal_val)
        {
            sim_env = start_state;
            int32_t curr_idx = root_idx;
            current_flipped = false;

            while (node_pool[curr_idx].is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float p_n = static_cast<float>(node_pool[curr_idx].n_visits);
                
                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

                for (int action : valids)
                {
                    int32_t child_idx = node_pool[curr_idx].children[action];
                    if (child_idx != -1)
                    {
                        float score = node_pool[child_idx].get_ucb(c_init, c_base, p_n, curr_idx == root_idx);
                        if (score > best_score) {
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

            int winner = sim_env.check_winner();
            if (winner != 0)
            {
                out_is_terminal = true;
                out_terminal_val = -1.0f;
                return;
            }
            else if (sim_env.is_full())
            {
                out_is_terminal = true;
                out_terminal_val = 0.0f;
                return;
            }

            out_is_terminal = false;

            std::mt19937 &rng = get_rng();
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            if (dist(rng) < 0.5f)
            {
                sim_env.flip();
                current_flipped = true;
            }
            else
            {
                current_flipped = false;
            }
            out_nn_input_board = sim_env;
        }

        void backprop(const std::vector<float> &policy_logits, float value, bool is_terminal)
        {
            if (current_leaf_idx == -1) return;

            // 终局状态不展开，保持为叶子节点（与 Python MCTS_AZ 行为一致）
            if (!is_terminal)
            {
                std::vector<float> final_policy = policy_logits;
                if (current_flipped)
                {
                    std::reverse(final_policy.begin(), final_policy.end());
                    sim_env.flip();
                }

                std::vector<int> valids = sim_env.get_valid_moves();

                std::vector<float> noise_vec;
                if (node_pool[current_leaf_idx].parent == -1 && alpha > 0.0f)  // 根节点
                {
                    std::mt19937& rng = get_rng();
                    std::gamma_distribution<float> gamma(alpha, 1.0f);

                    float sum = 0.0f;
                    for (size_t i = 0; i < valids.size(); ++i)
                    {
                        float n = gamma(rng);
                        noise_vec.push_back(n);
                        sum += n;
                    }
                    for (float &n : noise_vec)
                        n /= sum;
                }

                float policy_sum = 0.0f;
                for (int action : valids)
                {
                    policy_sum += final_policy[action];
                }

                // 展开逻辑
                int noise_idx = 0;
                for (int action : valids) {
                    float prob = final_policy[action] / (policy_sum + 1e-8f);
                    if (node_pool[current_leaf_idx].children[action] == -1) {
                        int32_t new_node = allocate_node(current_leaf_idx, prob);
                        node_pool[current_leaf_idx].children[action] = new_node;

                        // 设置 noise
                        if (node_pool[current_leaf_idx].parent == -1 && !noise_vec.empty()) {
                            node_pool[new_node].noise = noise_vec[noise_idx++];
                        }
                    }
                }
                node_pool[current_leaf_idx].is_expanded = true;
            }

            // 迭代式更新，替代递归
            int32_t update_idx = current_leaf_idx;
            float val = value;
            while (update_idx != -1) {
                node_pool[update_idx].n_visits++;
                node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
                val = -val * discount;
                update_idx = node_pool[update_idx].parent;
            }
        }

        std::vector<int> get_counts() const
        {
            std::vector<int> counts(Config::ACTION_SIZE, 0);
            for (int i = 0; i < Config::ACTION_SIZE; ++i)
            {
                int32_t child_idx = node_pool[root_idx].children[i];
                if (child_idx != -1)
                {
                    counts[i] = node_pool[child_idx].n_visits;
                }
            }
            return counts;
        }
    };
}

#endif /* D77F1FBF_0C41_461C_8809_93FD96ACA0C5 */