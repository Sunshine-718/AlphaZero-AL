/**
 * Connect4 游戏环境的 pybind11 绑定。
 *
 * 将 C++ Connect4 类包装为 Python Env 类，提供与旧 Cython/Python 环境完全兼容的 API：
 *   - board (property): 6×7 float32 ndarray（读/写）
 *   - turn (property): 当前落子方 1/-1
 *   - step(action), done(), winPlayer(), valid_move() 等游戏方法
 *   - current_state(): 返回 (1, 3, 6, 7) float32 张量
 *   - apply_symmetry(), random_symmetry(): 对称变换
 *   - pickle 支持（序列化/反序列化）
 *
 * 注册到 env_cpp.connect4 子模块。
 */
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

/// 将 C++ int8 内部棋盘导出为 6×7 float32 ndarray（copy）
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

/// 从 float32 ndarray 设置棋盘并重建 bitboard，自动推算 turn
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
    self.set_turn(self.n_pieces % 2 == 0 ? 1 : -1);
}

/// 从 board ndarray 构造 Connect4 实例（py::init 工厂）
inline Connect4 construct_from_board(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    Connect4 g;
    numpy_to_board(g, arr);
    return g;
}

// ── 注册 ─────────────────────────────────────────────────────────────────────

/// 将 Connect4 Env 注册到 env_cpp.connect4 子模块
inline void register_connect4(py::module_ &m)
{
    auto sub = m.def_submodule("connect4", "Connect4 environment");

    constexpr int R = Connect4::Traits::ROWS;
    constexpr int C = Connect4::Traits::COLS;
    constexpr int A = Connect4::Traits::ACTION_SIZE;

    py::class_<Connect4>(sub, "Env")

        // ── 构造 ─────────────────────────────────────────────────────────
        .def(py::init<>())                                          // 空棋盘
        .def(py::init(&construct_from_board), py::arg("board"))     // 从 ndarray 构造

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

        /// 重置为空棋盘
        .def("reset", &Connect4::reset)

        /// 深拷贝
        .def("copy", [](const Connect4 &self)
             { return Connect4(self); })

        /// 在指定列落子
        .def("step", &Connect4::step, py::arg("action"))

        /// 游戏是否结束（有人赢或平局）
        .def("done", [](const Connect4 &self)
             { return self.check_winner() != 0 || self.is_full(); })

        /// 获胜玩家：1 / -1 / 0（无人获胜）
        .def("winPlayer", &Connect4::check_winner)
        .def("check_winner", &Connect4::check_winner)

        /// 棋盘是否已满
        .def("check_full", &Connect4::is_full)

        /// 返回合法落子列号列表
        .def(
            "valid_move", [](const Connect4 &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                for (int i = 0; i < vm.count; ++i)
                    result.append(vm.moves[i]);
                return result; })

        /// 返回长度 7 的 bool 列表，表示各列是否可落子
        .def(
            "valid_mask", [](const Connect4 &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                bool mask[C] = {};
                for (int i = 0; i < vm.count; ++i)
                    mask[vm.moves[i]] = true;
                for (int c = 0; c < C; ++c)
                    result.append(mask[c]);
                return result; })

        /**
         * 返回 NN 输入张量 (1, 3, 6, 7) float32：
         *   通道 0 = 玩家 1 棋子位置
         *   通道 1 = 玩家 -1 棋子位置
         *   通道 2 = 当前落子方标识（全 1 或全 -1）
         */
        .def(
            "current_state", [](Connect4 &self)
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

        /**
         * 应用对称变换（水平翻转）。
         * @param sym_id  0=恒等, 1=水平翻转
         * @param inplace true=原地修改, false=返回副本
         */
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

        /// 对称逆变换动作：水平翻转下 col → COLS-1-col
        .def_static(
            "inverse_symmetry_action", [](int sym_id, int col)
            { return sym_id == 0 ? col : (C - 1 - col); },
            py::arg("sym_id"), py::arg("col"))

        /// 随机选择一个对称变换并应用，返回 (new_env, sym_id)
        .def(
            "random_symmetry", [](const Connect4 &self) -> py::tuple
            {
                static thread_local std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, Connect4::Traits::NUM_SYMMETRIES - 1);
                int sym_id = dist(rng);
                Connect4 copy(self);
                copy.apply_symmetry(sym_id);
                return py::make_tuple(copy, sym_id); })

        // ── Pickle ───────────────────────────────────────────────────────
        .def(py::pickle(
            // __getstate__: 序列化为 (board_ndarray, turn)
            [](Connect4 &self) -> py::tuple
            {
                return py::make_tuple(board_to_numpy(self), self.get_turn());
            },
            // __setstate__: 从 (board_ndarray, turn) 反序列化
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
