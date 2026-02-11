#ifndef A586E0F0_FE06_4A91_9BC8_B232FAB0344A
#define A586E0F0_FE06_4A91_9BC8_B232FAB0344A
#pragma once
#include <array>
#include <type_traits>

template <typename T>
concept MCTSGame = requires(T t, int action) {
    { t.reset() } -> std::same_as<void>;
    { t.get_valid_moves() };
    { t.step(action) } -> std::same_as<void>;
    { t.check_winner() } -> std::same_as<int>;
    { t.is_full() } -> std::same_as<bool>;
    { t.flip() } -> std::same_as<void>;

    std::is_copy_constructible_v<T>;
};

#endif /* A586E0F0_FE06_4A91_9BC8_B232FAB0344A */
