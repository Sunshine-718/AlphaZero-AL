#include <pybind11/pybind11.h>

#include "env_connect4.h"
// 添加新游戏时在这里 include，例如:
// #include "env_tictactoe.h"

namespace py = pybind11;

PYBIND11_MODULE(env_cpp, m)
{
    m.doc() = "Game environments (C++ bitboard backends)";

    // === 注册各游戏的 Env ===
    env_bind::register_connect4(m);

    // 添加新游戏只需一行:
    // env_bind::register_tictactoe(m);
    // env_bind::register_gomoku(m);
}
