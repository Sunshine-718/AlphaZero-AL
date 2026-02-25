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
    inline std::mt19937 &get_rng()
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        return rng;
    }

    template <MCTSGame Game>
    class MCTS
    {
    public:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
        using Node = MCTSNode<ACTION_SIZE>;

        // 节点池
        std::vector<Node> node_pool;
        int32_t root_idx = 0;
        int32_t next_free_node = 0;

        Game sim_env;
        Game leaf_state;
        int32_t current_leaf_idx = -1;
        int current_sym_id = 0;

        float c_init, c_base, alpha;
        float noise_epsilon;
        float fpu_reduction;
        bool use_symmetry;
        float mlh_slope;
        float mlh_cap;

        MCTS(float c_i, float c_b, float a, float noise_eps = 0.25f, float fpu_red = 0.4f, bool use_sym = true,
             float mlh_slope_ = 0.0f, float mlh_cap_ = 0.2f)
            : c_init(c_i), c_base(c_b), alpha(a), noise_epsilon(noise_eps), fpu_reduction(fpu_red), use_symmetry(use_sym),
              mlh_slope(mlh_slope_), mlh_cap(mlh_cap_)
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

        void prune_root(int action)
        {
            int32_t child_idx = node_pool[root_idx].children[action];
            if (action >= 0 && action < ACTION_SIZE && child_idx != -1)
            {
                root_idx = child_idx;
                node_pool[root_idx].parent = -1;
                // 为新 root 的子节点刷新 Dirichlet 噪声
                apply_root_noise();
            } else
            {
                reset();
            }
        }

        // 为当前 root 的所有已有子节点重新生成 Dirichlet 噪声
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

        void simulate(const Game &start_state, Game &out_nn_input_board, bool &out_is_terminal, float &out_terminal_val)
        {
            sim_env = start_state;
            int32_t curr_idx = root_idx;
            current_sym_id = 0;

            while (node_pool[curr_idx].is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float p_n = static_cast<float>(node_pool[curr_idx].n_visits);

                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

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
                // 劣势时动态降低 fpu_reduction，鼓励探索新走法
                // Q ∈ [-1,1] → scale ∈ [0,1]; Q=+1→1(保守) Q=-1→0(全面探索)
                float scale = (1.0f + parent_value) / 2.0f;
                float effective_fpu = fpu_reduction * scale;
                float fpu_value = parent_value - effective_fpu * std::sqrt(seen_policy);
                fpu_value = std::max(-1.0f, fpu_value);

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

            int winner = sim_env.check_winner();
            if (winner != 0)
            {
                out_is_terminal = true;
                out_terminal_val = -1.0f;
                out_nn_input_board = sim_env;
                return;
            } else if (sim_env.is_full())
            {
                out_is_terminal = true;
                out_terminal_val = 0.0f;
                out_nn_input_board = sim_env;
                return;
            }

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

        void backprop(std::span<const float> policy_logits, float value, float moves_left, bool is_terminal)
        {
            if (current_leaf_idx == -1) return;

            // 终局状态不展开，保持为叶子节点（与 Python MCTS_AZ 行为一致）
            if (!is_terminal)
            {
                std::array<float, ACTION_SIZE> final_policy;
                std::copy(policy_logits.begin(), policy_logits.end(), final_policy.begin());
                Game::inverse_symmetry_policy(current_sym_id, final_policy);

                auto valids = leaf_state.get_valid_moves();

                // Dirichlet 噪声（栈上数组，避免堆分配）
                float noise_arr[ACTION_SIZE];
                int noise_count = 0;
                if (node_pool[current_leaf_idx].parent == -1 && alpha > 0.0f)  // 根节点
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

                float policy_sum = 0.0f;
                for (int action : valids)
                {
                    policy_sum += final_policy[action];
                }

                // 展开逻辑
                int noise_idx = 0;
                for (int action : valids)
                {
                    float prob = final_policy[action] / (policy_sum + 1e-8f);
                    if (node_pool[current_leaf_idx].children[action] == -1)
                    {
                        int32_t new_node = allocate_node(current_leaf_idx, prob);
                        node_pool[current_leaf_idx].children[action] = new_node;

                        // 设置 noise
                        if (noise_count > 0)
                        {
                            node_pool[new_node].noise = noise_arr[noise_idx++];
                        }
                    }
                }
                node_pool[current_leaf_idx].is_expanded = true;
            }

            // 迭代式更新，替代递归
            int32_t update_idx = current_leaf_idx;
            float val = value;
            float ml = is_terminal ? 0.0f : moves_left;
            while (update_idx != -1)
            {
                node_pool[update_idx].n_visits++;
                node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
                node_pool[update_idx].M += (ml - node_pool[update_idx].M) / node_pool[update_idx].n_visits;
                val = -val;
                ml += 1.0f;  // 父节点比子节点多一步
                update_idx = node_pool[update_idx].parent;
            }
        }

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
    };
}

#endif /* D77F1FBF_0C41_461C_8809_93FD96ACA0C5 */
