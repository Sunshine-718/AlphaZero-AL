#ifndef A586E0F0_FE06_4A91_9BC8_B232FAB0344A
#define A586E0F0_FE06_4A91_9BC8_B232FAB0344A
#pragma once
#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace AlphaZero
{
    // 栈上固定大小的合法动作列表，避免 std::vector 堆分配
    template <int MAX_ACTIONS>
    struct ValidMoves
    {
        int moves[MAX_ACTIONS];
        int count = 0;

        [[nodiscard]] bool empty() const { return count == 0; }
        [[nodiscard]] int size() const { return count; }
        const int *begin() const { return moves; }
        const int *end() const { return moves + count; }
    };

    // 游戏类型必须满足的 concept，MCTS 层通过此接口与游戏交互
    template <typename G>
    concept MCTSGame = requires(G g, const G cg, int action, const int8_t *src) {
        // --- 编译期 Traits ---
        { G::Traits::ACTION_SIZE } -> std::convertible_to<int>;
        { G::Traits::BOARD_SIZE } -> std::convertible_to<int>;
        { G::Traits::NUM_SYMMETRIES } -> std::convertible_to<int>;

        // --- 核心游戏逻辑 ---
        { g.reset() } -> std::same_as<void>;
        { g.step(action) } -> std::same_as<void>;
        { cg.check_winner() } -> std::same_as<int>;
        { cg.is_full() } -> std::same_as<bool>;
        { cg.get_valid_moves() };

        // --- 对称变换 ---
        { g.apply_symmetry(action) } -> std::same_as<void>;

        // --- Python memcpy I/O ---
        { cg.board_data() } -> std::same_as<const int8_t *>;
        { cg.get_turn() } -> std::same_as<int>;
        { g.set_turn(action) } -> std::same_as<void>;
        { g.import_board(src) } -> std::same_as<void>;

        // 可拷贝
        requires std::is_copy_constructible_v<G>;
    } && requires(std::array<float, G::Traits::ACTION_SIZE> &policy) {
        // static: 对 policy 数组应用对称逆变换
        { G::inverse_symmetry_policy(0, policy) } -> std::same_as<void>;
    };
}

#endif /* A586E0F0_FE06_4A91_9BC8_B232FAB0344A */
