/**
 * Othello 游戏环境的 pybind11 绑定。
 *
 * 公共方法（board I/O, current_state, apply_symmetry 等）由 env_common.h 提供。
 * 本文件仅包含 Othello 特有的绑定：构造器、done、show、inverse_symmetry_action。
 *
 * 注册到 env_cpp.othello 子模块。
 */
#pragma once

#include "env_common.h"
#include "Othello.h"
#include <sstream>

using namespace AlphaZero;

namespace env_bind
{

inline void register_othello(py::module_ &m)
{
    auto sub = m.def_submodule("othello", "Othello environment");

    constexpr int R = Othello::Traits::ROWS;
    constexpr int C = Othello::Traits::COLS;

    py::class_<Othello> cls(sub, "Env");

    // ── 构造 ─────────────────────────────────────────────────────────
    cls.def(py::init<>())
       .def(py::init(&construct_from_board<Othello>), py::arg("board"));

    // ── 公共方法 ─────────────────────────────────────────────────────
    register_env_common(cls, Othello::Traits::NUM_SYMMETRIES);
    register_env_traits<Othello>(cls);

    // ── Othello 特有方法 ─────────────────────────────────────────────
    cls
        .def("done", [](const Othello &self)
             { return self.is_game_over(); })

        .def_static(
            "inverse_symmetry_action", [](int sym_id, int action) -> int
            {
                if (action == Othello::PASS_ACTION || sym_id == 0)
                    return action;
                int inv = Othello::inverse_sym(sym_id);
                int nr, nc;
                Othello::transform_coord(inv, action / 8, action % 8, nr, nc);
                return nr * 8 + nc;
            },
            py::arg("sym_id"), py::arg("action"))

        .def(
            "show", [](Othello &self)
            {
                self.sync_to_board();
                std::ostringstream os;
                os << "========================\n";
                os << "  0 1 2 3 4 5 6 7\n";
                for (int r = 0; r < R; ++r)
                {
                    os << r << ' ';
                    for (int c = 0; c < C; ++c)
                    {
                        if (c > 0) os << ' ';
                        int8_t v = self.board[r][c];
                        os << (v == 0 ? '.' : (v == 1 ? 'X' : 'O'));
                    }
                    os << '\n';
                }
                os << "========================";
                py::print(os.str()); });
}

} // namespace env_bind
