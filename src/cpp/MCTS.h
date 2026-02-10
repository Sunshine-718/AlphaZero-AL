#ifndef D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#define D77F1FBF_0C41_461C_8809_93FD96ACA0C5
#pragma once
#include "Connect4.h"
#include "MCTSNode.h"
#include <optional>
#include <random>

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
        std::unique_ptr<MCTSNode> root;

        Connect4 sim_env;

        MCTSNode *current_leaf = nullptr;
        bool current_flipped = false;

        float c_init, c_base, discount, alpha;

        MCTS(float c_i, float c_b, float disc, float a)
            : c_init(c_i), c_base(c_b), discount(disc), alpha(a)
        {
            reset();
        }

        void reset()
        {
            root = std::make_unique<MCTSNode>(nullptr, 1.0f);
            root->is_expanded = false;
        }

        void prune_root(int action)
        {
            if (action >= 0 && action < Config::ACTION_SIZE && root->children[action])
            {
                std::unique_ptr<MCTSNode> new_root = std::move(root->children[action]);
                root = std::move(new_root);
                root->parent = nullptr;
            }
            else
            {
                reset();
            }
        }

        void simulate(const Connect4 &start_state, Connect4 &out_nn_input_board, bool &out_is_terminal, float &out_terminal_val)
        {
            sim_env = start_state;
            MCTSNode *node = root.get();
            current_flipped = false;

            while (node->is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float parent_n = static_cast<float>(node->n_visits);
                std::vector<int> valids = sim_env.get_valid_moves();

                if (valids.empty())
                    break;

                for (int action : valids)
                {
                    if (node->children[action])
                    {
                        float score = node->children[action]->get_ucb(c_init, c_base, parent_n);
                        if (score > best_score)
                        {
                            best_score = score;
                            best_action = action;
                        }
                    }
                }
                if (best_action == -1)
                    break;

                sim_env.step(best_action);
                node = node->children[best_action].get();

                if (sim_env.check_winner() != 0 || sim_env.is_full())
                {
                    break;
                }
            }
            current_leaf = node;

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

        void backprop(const std::vector<float> &policy_logits, float value)
        {
            if (!current_leaf)
                return;

            std::vector<float> final_policy = policy_logits;
            if (current_flipped)
            {
                std::reverse(final_policy.begin(), final_policy.end());
                sim_env.flip();
            }

            std::vector<int> valids = sim_env.get_valid_moves();

            std::vector<float> noise_vec;
            if (current_leaf->is_root() && alpha > 0.0f)
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

            int noise_idx = 0;
            for (int action : valids)
            {
                float prob = final_policy[action] / (policy_sum + 1e-8f);

                if (!current_leaf->children[action])
                {
                    current_leaf->children[action] = std::make_unique<MCTSNode>(current_leaf, prob);

                    if (current_leaf->is_root() && !noise_vec.empty())
                    {
                        current_leaf->children[action]->noise = noise_vec[noise_idx++];
                    }
                }
            }
            current_leaf->is_expanded = true;

            current_leaf->update(value, discount);
        }

        std::vector<int> get_counts() const
        {
            std::vector<int> counts(Config::ACTION_SIZE, 0);
            for (int i = 0; i < Config::ACTION_SIZE; ++i)
            {
                if (root->children[i])
                {
                    counts[i] = root->children[i]->n_visits;
                }
            }
            return counts;
        }
    };
}

#endif /* D77F1FBF_0C41_461C_8809_93FD96ACA0C5 */
