#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Connect4.h"
#include <cstring>
#include <random>
#include <sstream>

namespace py = pybind11;
using namespace AlphaZero;

namespace env_bind {

// ── helpers ──────────────────────────────────────────────────────────────────

// board (int8 内部) → 6×7 float32 ndarray (copy)
inline py::array_t<float> board_to_numpy(Connect4 &self)
{
    self.sync_to_board();
    constexpr int R = Connect4::Traits::ROWS;
    constexpr int C = Connect4::Traits::COLS;
    py::array_t<float> arr({R, C});
    auto buf = arr.mutable_unchecked<2>();
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            buf(r, c) = static_cast<float>(self.board[r][c]);
    return arr;
}

// float32 ndarray → 内部 int8 board + 重建 bitboard
inline void numpy_to_board(Connect4 &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    constexpr int R = Connect4::Traits::ROWS;
    constexpr int C = Connect4::Traits::COLS;
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != R || buf.shape(1) != C)
        throw std::runtime_error("board shape must be (6, 7)");
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            self.board[r][c] = static_cast<int8_t>(buf(r, c));
    self.sync_from_board();
    // 根据棋子数推算 turn
    self.set_turn(self.n_pieces % 2 == 0 ? 1 : -1);
}

// 从 board 构造
inline Connect4 construct_from_board(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    Connect4 g;
    numpy_to_board(g, arr);
    return g;
}

// ── 注册 ─────────────────────────────────────────────────────────────────────

inline void register_connect4(py::module_ &m)
{
    auto sub = m.def_submodule("connect4", "Connect4 environment");

    constexpr int R = Connect4::Traits::ROWS;
    constexpr int C = Connect4::Traits::COLS;
    constexpr int A = Connect4::Traits::ACTION_SIZE;

    py::class_<Connect4>(sub, "Env")

        // ── 构造 ─────────────────────────────────────────────────────────
        .def(py::init<>()) // 默认空棋盘
        .def(py::init(&construct_from_board), py::arg("board"))

        // ── Properties ───────────────────────────────────────────────────
        .def_property("board", &board_to_numpy, &numpy_to_board)

        .def_property(
            "turn",
            &Connect4::get_turn,
            &Connect4::set_turn)

        .def_property_readonly_static(
            "NUM_SYMMETRIES",
            [](py::object) { return Connect4::Traits::NUM_SYMMETRIES; })

        // ── 核心方法 ─────────────────────────────────────────────────────
        .def("reset", &Connect4::reset)

        .def("copy", [](const Connect4 &self)
             { return Connect4(self); })

        .def("step", &Connect4::step, py::arg("action"))

        .def("done", [](const Connect4 &self)
             { return self.check_winner() != 0 || self.is_full(); })

        .def("winPlayer", &Connect4::check_winner)
        .def("check_winner", &Connect4::check_winner)
        .def("check_full", &Connect4::is_full)

        .def(
            "valid_move", [](const Connect4 &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                for (int i = 0; i < vm.count; ++i)
                    result.append(vm.moves[i]);
                return result; })

        .def(
            "valid_mask", [](const Connect4 &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                // 构造长度 = COLS 的 bool 列表
                bool mask[C] = {};
                for (int i = 0; i < vm.count; ++i)
                    mask[vm.moves[i]] = true;
                for (int c = 0; c < C; ++c)
                    result.append(mask[c]);
                return result; })

        // ── current_state: (1, 3, 6, 7) float32 ─────────────────────────
        .def(
            "current_state", [](Connect4 &self)
            {
                self.sync_to_board();
                py::array_t<float> state({1, 3, R, C});
                auto buf = state.mutable_unchecked<4>();
                // 全部清零
                std::memset(buf.mutable_data(0, 0, 0, 0), 0, sizeof(float) * 3 * R * C);
                float turn_f = static_cast<float>(self.get_turn());
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c)
                    {
                        int8_t v = self.board[r][c];
                        if (v == 1)
                            buf(0, 0, r, c) = 1.0f;
                        else if (v == -1)
                            buf(0, 1, r, c) = 1.0f;
                        buf(0, 2, r, c) = turn_f;
                    }
                return state; })

        // ── show ─────────────────────────────────────────────────────────
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
                py::print(os.str()); })

        // ── 对称变换 ─────────────────────────────────────────────────────
        .def(
            "apply_symmetry", [](const Connect4 &self, int sym_id, bool inplace) -> Connect4
            {
                if (inplace)
                {
                    Connect4 &mut_self = const_cast<Connect4 &>(self);
                    mut_self.apply_symmetry(sym_id);
                    return mut_self;
                }
                Connect4 copy(self);
                copy.apply_symmetry(sym_id);
                return copy; },
            py::arg("sym_id"), py::arg("inplace") = false)

        .def_static(
            "inverse_symmetry_action", [](int sym_id, int col)
            { return sym_id == 0 ? col : (C - 1 - col); },
            py::arg("sym_id"), py::arg("col"))

        .def(
            "random_symmetry", [](const Connect4 &self) -> py::tuple
            {
                // 使用 thread_local RNG
                static thread_local std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, Connect4::Traits::NUM_SYMMETRIES - 1);
                int sym_id = dist(rng);
                Connect4 copy(self);
                copy.apply_symmetry(sym_id);
                return py::make_tuple(copy, sym_id); })

        // ── Pickle ───────────────────────────────────────────────────────
        .def(py::pickle(
            // __getstate__
            [](Connect4 &self) -> py::tuple
            {
                return py::make_tuple(board_to_numpy(self), self.get_turn());
            },
            // __setstate__
            [](py::tuple t) -> Connect4
            {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid pickle state");
                Connect4 g;
                numpy_to_board(g, t[0].cast<py::array_t<float>>());
                g.set_turn(t[1].cast<int>());
                return g;
            }));
}

} // namespace env_bind
