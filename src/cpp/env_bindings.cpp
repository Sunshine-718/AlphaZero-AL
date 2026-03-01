/**
 * 游戏环境 pybind11 绑定聚合模块。
 *
 * 将各游戏的 Env 类注册为 env_cpp 模块的子模块：
 *   env_cpp.connect4.Env  — Connect4 环境
 *
 * 添加新游戏：#include 对应头文件，调用注册函数。
 */
#include <pybind11/pybind11.h>

#include "env_connect4.h"
#include "env_othello.h"

namespace py = pybind11;

PYBIND11_MODULE(env_cpp, m)
{
    m.doc() = "Game environments (C++ bitboard backends)";

    // === 注册各游戏的 Env ===
    env_bind::register_connect4(m);
    env_bind::register_othello(m);
}
