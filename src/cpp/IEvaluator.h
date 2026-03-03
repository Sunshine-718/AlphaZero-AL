#pragma once
#include "GameContext.h"
#include "MCTSNode.h" // for WDLValue
#include <array>
#include <omp.h>
#include <span>
#include <vector>

namespace AlphaZero
{
    /**
     * 叶节点评估器抽象接口。
     *
     * 将评估策略从 MCTS 核心中解耦，支持不同评估方式（rollout、NN、NNUE 等）。
     * 虚函数开销可忽略：evaluate 在 batch 粒度调用（~800 次/搜索），
     * 远低于游戏逻辑的 ~12M 次/s 调用频率。
     *
     * @tparam Game 满足 MCTSGame concept 的游戏类型
     */
    template <MCTSGame Game>
    class IEvaluator
    {
    public:
        static constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;

        /// 单个叶节点的评估结果
        struct Result
        {
            std::array<float, ACTION_SIZE> policy;
            WDLValue wdl;
            float moves_left = 0.0f;
        };

        virtual ~IEvaluator() = default;

        /**
         * 批量评估叶节点。
         * 默认实现：逐个调 evaluate_single()，OpenMP 并行。
         * NN 评估器可重写此方法实现真正的 batch 推理。
         */
        virtual void evaluate_batch(
            std::span<const Game> states,
            std::span<const int> turns,
            std::span<Result> results)
        {
            int n = static_cast<int>(states.size());
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                results[i] = evaluate_single(states[i], turns[i]);
        }

        /**
         * 评估单个叶节点。
         * 简单评估器（如 rollout）重写此方法即可，batch 版本自动并行调用。
         */
        virtual Result evaluate_single(const Game &state, int turn)
        {
            // 默认：均匀 policy + 均匀 WDL
            Result r;
            r.policy.fill(1.0f);
            r.wdl = WDLValue::uniform();
            r.moves_left = 0.0f;
            return r;
        }
    };
}
