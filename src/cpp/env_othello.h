/**
 * Othello 游戏环境的 pybind11 绑定。
 *
 * 将 C++ Othello 类包装为 Python Env 类，提供与 Connect4 环境兼容的 API：
 *   - board (property): 8×8 float32 ndarray（读/写）
 *   - turn (property): 当前落子方 1/-1
 *   - step(action), done(), winPlayer(), valid_move() 等游戏方法
 *   - current_state(): 返回 (1, 3, 8, 8) float32 张量
 *   - apply_symmetry(), random_symmetry(): 对称变换
 *   - pickle 支持（序列化/反序列化）
 *
 * 注册到 env_cpp.othello 子模块。
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Othello.h"
#include <cstring>
#include <random>
#include <sstream>

namespace py = pybind11;
using namespace AlphaZero;

namespace env_bind {

// ── helpers ──────────────────────────────────────────────────────────────────

/// 将 C++ int8 内部棋盘导出为 8×8 float32 ndarray（copy）
inline py::array_t<float> othello_board_to_numpy(Othello &self)
{
    self.sync_to_board();
    constexpr int R = Othello::Traits::ROWS;
    constexpr int C = Othello::Traits::COLS;
    py::array_t<float> arr({R, C});
    auto buf = arr.mutable_unchecked<2>();
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            buf(r, c) = static_cast<float>(self.board[r][c]);
    return arr;
}

/// 从 float32 ndarray 设置棋盘并重建 bitboard
inline void othello_numpy_to_board(Othello &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    constexpr int R = Othello::Traits::ROWS;
    constexpr int C = Othello::Traits::COLS;
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != R || buf.shape(1) != C)
        throw std::runtime_error("board shape must be (8, 8)");
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            self.board[r][c] = static_cast<int8_t>(buf(r, c));
    self.sync_from_board();
    // 启发式推断 turn：无 pass 时正确，有 pass 时可能出错
    // 关键路径（search_batch、pickle）会在之后显式 set_turn() 覆盖
    self.set_turn(self.n_pieces % 2 == 0 ? 1 : -1);
}

/// 从 board ndarray 构造 Othello 实例
inline Othello othello_construct_from_board(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    Othello g;
    othello_numpy_to_board(g, arr);
    return g;
}

// ── 注册 ─────────────────────────────────────────────────────────────────────

/// 将 Othello Env 注册到 env_cpp.othello 子模块
inline void register_othello(py::module_ &m)
{
    auto sub = m.def_submodule("othello", "Othello environment");

    constexpr int R = Othello::Traits::ROWS;
    constexpr int C = Othello::Traits::COLS;
    constexpr int A = Othello::Traits::ACTION_SIZE;

    py::class_<Othello>(sub, "Env")

        // ── 构造 ─────────────────────────────────────────────────────────
        .def(py::init<>())
        .def(py::init(&othello_construct_from_board), py::arg("board"))

        // ── Properties ───────────────────────────────────────────────────
        .def_property("board", &othello_board_to_numpy, &othello_numpy_to_board)

        .def_property(
            "turn",
            &Othello::get_turn,
            &Othello::set_turn)

        .def_property_readonly_static(
            "NUM_SYMMETRIES",
            [](py::object) { return Othello::Traits::NUM_SYMMETRIES; })

        // ── 核心方法 ─────────────────────────────────────────────────────

        .def("reset", &Othello::reset)

        .def("copy", [](const Othello &self)
             { return Othello(self); })

        .def("step", &Othello::step, py::arg("action"))

        .def("done", [](const Othello &self)
             { return self.is_game_over(); })

        .def("winPlayer", &Othello::check_winner)
        .def("check_winner", &Othello::check_winner)

        .def("check_full", &Othello::is_full)

        /// 返回合法动作列表（0-63 为放置，64 为 pass）
        .def(
            "valid_move", [](const Othello &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                for (int i = 0; i < vm.count; ++i)
                    result.append(vm.moves[i]);
                return result; })

        /// 返回长度 65 的 bool 列表，表示各动作是否合法
        .def(
            "valid_mask", [](const Othello &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                bool mask[A] = {};
                for (int i = 0; i < vm.count; ++i)
                    mask[vm.moves[i]] = true;
                for (int a = 0; a < A; ++a)
                    result.append(mask[a]);
                return result; })

        /**
         * 返回 NN 输入张量 (1, 3, 8, 8) float32：
         *   通道 0 = Black (P1) 棋子位置
         *   通道 1 = White (P2) 棋子位置
         *   通道 2 = 当前落子方标识（全 1 或全 -1）
         */
        .def(
            "current_state", [](Othello &self)
            {
                self.sync_to_board();
                py::array_t<float> state({1, 3, R, C});
                auto buf = state.mutable_unchecked<4>();
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

        /// 打印棋盘到控制台
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
                py::print(os.str()); })

        // ── 对称变换 ─────────────────────────────────────────────────────

        .def(
            "apply_symmetry", [](const Othello &self, int sym_id, bool inplace) -> Othello
            {
                if (inplace)
                {
                    Othello &mut_self = const_cast<Othello &>(self);
                    mut_self.apply_symmetry(sym_id);
                    return mut_self;
                }
                Othello copy(self);
                copy.apply_symmetry(sym_id);
                return copy; },
            py::arg("sym_id"), py::arg("inplace") = false)

        /// 对称逆变换动作
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

        /// 随机选择一个对称变换并应用，返回 (new_env, sym_id)
        .def(
            "random_symmetry", [](const Othello &self) -> py::tuple
            {
                static thread_local std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, Othello::Traits::NUM_SYMMETRIES - 1);
                int sym_id = dist(rng);
                Othello copy(self);
                copy.apply_symmetry(sym_id);
                return py::make_tuple(copy, sym_id); })

        // ── Pickle ───────────────────────────────────────────────────────
        .def(py::pickle(
            [](Othello &self) -> py::tuple
            {
                return py::make_tuple(othello_board_to_numpy(self), self.get_turn());
            },
            [](py::tuple t) -> Othello
            {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid pickle state");
                Othello g;
                othello_numpy_to_board(g, t[0].cast<py::array_t<float>>());
                g.set_turn(t[1].cast<int>());
                return g;
            }));
}

} // namespace env_bind
