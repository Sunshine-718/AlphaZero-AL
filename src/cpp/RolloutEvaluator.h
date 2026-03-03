#pragma once
#include "IEvaluator.h"
#include <random>

namespace AlphaZero
{
    /// 前向声明：线程本地 RNG（定义在 MCTS.h）
    inline std::mt19937 &get_rng();

    /**
     * Random Rollout 评估器：对叶节点随机模拟到终局，返回均匀 policy + 终局 WDL。
     * 用于纯 MCTS 基线，替代神经网络评估。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class RolloutEvaluator : public IEvaluator<Game>
    {
    public:
        using Result = typename IEvaluator<Game>::Result;
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;

        Result evaluate_single(const Game &state, int turn) override
        {
            Result r;
            r.policy.fill(1.0f); // uniform prior
            r.moves_left = 0.0f;

            // Random rollout to terminal
            Game sim = state;
            while (true)
            {
                int w = sim.check_winner();
                if (w != 0)
                {
                    r.wdl = winner_to_wdl(w);
                    return r;
                }
                if (sim.is_full())
                {
                    r.wdl = WDLValue::draw();
                    return r;
                }
                auto valids = sim.get_valid_moves();
                std::uniform_int_distribution<int> dist(0, valids.size() - 1);
                sim.step(valids.moves[dist(get_rng())]);
            }
        }
    };
}
