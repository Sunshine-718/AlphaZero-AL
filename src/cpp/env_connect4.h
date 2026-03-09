/**
 * Connect4 游戏环境的 pybind11 绑定。
 *
 * 公共方法（board I/O, current_state, apply_symmetry 等）由 env_common.h 提供。
 * 本文件仅包含 Connect4 特有的绑定：构造器、done、show、inverse_symmetry_action。
 *
 * 注册到 env_cpp.connect4 子模块。
 */
#pragma once

#include "env_common.h"
#include "Connect4.h"
#include <sstream>

using namespace AlphaZero;

namespace env_bind
{

inline void register_connect4(py::module_ &m)
{
    auto sub = m.def_submodule("connect4", "Connect4 environment");

    constexpr int R = Connect4::Traits::ROWS;
    constexpr int C = Connect4::Traits::COLS;

    py::class_<Connect4> cls(sub, "Env");

    // ── 构造 ─────────────────────────────────────────────────────────
    cls.def(py::init<>())
       .def(py::init(&construct_from_board<Connect4>), py::arg("board"));

    // ── 公共方法 ─────────────────────────────────────────────────────
    register_env_common(cls, Connect4::Traits::NUM_SYMMETRIES);
    register_env_traits<Connect4>(cls);

    // ── Connect4 特有方法 ────────────────────────────────────────────
    cls
        .def("done", [](const Connect4 &self)
             { return self.check_winner() != 0 || self.is_full(); })

        .def_static(
            "inverse_symmetry_action", [](int sym_id, int col)
            { return sym_id == 0 ? col : (C - 1 - col); },
            py::arg("sym_id"), py::arg("col"))

        .def(
            "show", [](Connect4 &self)
            {
                self.sync_to_board();
                std::ostringstream os;
                os << "====================\n";
                for (int r = 0; r < R; ++r)
                {
                    for (int c = 0; c < C; ++c)
                    {
                        if (c > 0) os << ' ';
                        int8_t v = self.board[r][c];
                        os << (v == 0 ? '_' : (v == 1 ? 'X' : 'O'));
                    }
                    os << '\n';
                }
                os << "0 1 2 3 4 5 6\n";
                os << "====================";
                py::print(os.str()); });
}

} // namespace env_bind
