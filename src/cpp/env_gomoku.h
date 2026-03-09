/**
 * Gomoku 游戏环境的 pybind11 绑定。
 *
 * 公共方法（reset, copy, step, apply_symmetry 等）由 env_common.h 提供。
 * Gomoku 使用运行时维度，因此 board I/O、valid_move/mask、current_state、pickle
 * 仍为 Gomoku 特有实现。
 *
 * 注册到 env_cpp.gomoku 子模块。
 */
#pragma once

#include "env_common.h"
#include "Gomoku.h"
#include <sstream>
#include <string>

using namespace AlphaZero;

namespace env_bind
{

// ── Gomoku 特有的 Board I/O 辅助 ────────────────────────────────────────

inline py::array_t<float> gomoku_board_to_numpy(const Gomoku &self)
{
    py::array_t<float> arr({self.rows(), self.cols()});
    auto buf = arr.mutable_unchecked<2>();
    const int8_t *src = self.board_data();
    for (int r = 0; r < self.rows(); ++r)
        for (int c = 0; c < self.cols(); ++c)
            buf(r, c) = static_cast<float>(src[r * self.cols() + c]);
    return arr;
}

inline void gomoku_numpy_to_board(Gomoku &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != self.rows() || buf.shape(1) != self.cols())
        throw std::runtime_error("board shape does not match environment dimensions");

    std::vector<int8_t> tmp(static_cast<size_t>(self.action_size()), 0);
    for (int r = 0; r < self.rows(); ++r)
        for (int c = 0; c < self.cols(); ++c)
            tmp[r * self.cols() + c] = static_cast<int8_t>(buf(r, c));
    self.import_board(tmp.data());
}

inline Gomoku gomoku_construct_from_board(py::array_t<float, py::array::c_style | py::array::forcecast> arr, int n_in_row)
{
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != buf.shape(1))
        throw std::runtime_error("board must be square");
    Gomoku g(static_cast<int>(buf.shape(0)), n_in_row);
    gomoku_numpy_to_board(g, arr);
    return g;
}

// ── 注册 ─────────────────────────────────────────────────────────────────

inline void register_gomoku(py::module_ &m)
{
    auto sub = m.def_submodule("gomoku", "Gomoku-like environment with configurable board and win length");

    py::class_<Gomoku> cls(sub, "Env");

    // ── 构造 ─────────────────────────────────────────────────────────
    cls.def(py::init<int, int>(),
            py::arg("board_size") = 15,
            py::arg("n_in_row") = 5)
       .def(py::init(&gomoku_construct_from_board),
            py::arg("board"),
            py::arg("n_in_row") = 5);

    // ── 公共方法（reset, copy, step, check_winner, apply_symmetry 等）
    register_env_common(cls, Gomoku::NUM_SYMMETRIES);

    // ── Gomoku 特有绑定 ──────────────────────────────────────────────
    cls
        .def_property("board", &gomoku_board_to_numpy, &gomoku_numpy_to_board)

        .def_property_readonly("board_size", &Gomoku::board_size)
        .def_property_readonly("rows", &Gomoku::rows)
        .def_property_readonly("cols", &Gomoku::cols)
        .def_property_readonly("n_in_row", &Gomoku::n_in_row)
        .def_property_readonly("action_size", &Gomoku::action_size)
        .def_property_readonly("num_symmetries", &Gomoku::num_symmetries)

        .def("set_params", &Gomoku::set_params,
             py::arg("board_size"), py::arg("n_in_row"))

        .def("done", &Gomoku::is_done)

        .def("step_xy", [](Gomoku &self, int row, int col)
             { self.step(self.coord_to_action(row, col)); },
             py::arg("row"), py::arg("col"))

        .def("coord_to_action", &Gomoku::coord_to_action,
             py::arg("row"), py::arg("col"))

        .def("action_to_coord", [](const Gomoku &self, int action)
             {
                 if (action < 0 || action >= self.action_size())
                     throw std::runtime_error("action out of range");
                 return py::make_tuple(action / self.cols(), action % self.cols());
             },
             py::arg("action"))

        .def("valid_move", [](const Gomoku &self)
             {
                 py::list result;
                 auto moves = self.get_valid_moves();
                 for (int a : moves)
                     result.append(a);
                 return result;
             })

        .def("valid_mask", [](const Gomoku &self)
             {
                 py::list result;
                 const int8_t *b = self.board_data();
                 for (int i = 0; i < self.action_size(); ++i)
                     result.append(b[i] == 0);
                 return result;
             })

        .def("current_state", [](Gomoku &self)
             { return make_current_state(self, self.rows(), self.cols()); })

        .def("inverse_symmetry_action", &Gomoku::inverse_symmetry_action,
             py::arg("sym_id"), py::arg("action"))

        .def("show", [](const Gomoku &self)
             {
                 const int8_t *b = self.board_data();
                 std::ostringstream os;
                 os << "==============================\n";
                 os << "    ";
                 for (int c = 0; c < self.cols(); ++c)
                     os << c % 10 << ' ';
                 os << '\n';
                 for (int r = 0; r < self.rows(); ++r)
                 {
                     os << (r < 10 ? " " : "") << r << "  ";
                     for (int c = 0; c < self.cols(); ++c)
                     {
                         const int8_t v = b[r * self.cols() + c];
                         os << (v == 0 ? '.' : (v == 1 ? 'X' : 'O')) << ' ';
                     }
                     os << '\n';
                 }
                 os << "==============================";
                 py::print(os.str());
             })

        // Pickle: (board_ndarray, turn, n_in_row)
        .def(py::pickle(
            [](const Gomoku &self) -> py::tuple
            {
                return py::make_tuple(gomoku_board_to_numpy(self), self.get_turn(), self.n_in_row());
            },
            [](py::tuple t) -> Gomoku
            {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid pickle state");
                auto board = t[0].cast<py::array_t<float>>();
                int n_in_row = t[2].cast<int>();
                Gomoku g = gomoku_construct_from_board(board, n_in_row);
                g.set_turn(t[1].cast<int>());
                return g;
            }));
}

} // namespace env_bind
