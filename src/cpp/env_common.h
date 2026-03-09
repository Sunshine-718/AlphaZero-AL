/**
 * 游戏环境 pybind11 绑定的公共模板工具。
 *
 * 提供：
 *   1. Board I/O 辅助函数（Traits 游戏通用）
 *   2. NN 输入张量构建（所有游戏通用）
 *   3. register_env_common()  — 注册所有游戏共享的 pybind11 方法
 *   4. register_env_traits()  — 注册 Traits 游戏（编译期维度）的额外公共方法
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <concepts>
#include <cstring>
#include <random>

namespace py = pybind11;

namespace env_bind
{

// ── Concepts ────────────────────────────────────────────────────────────────

/// 检测游戏是否有 sync_to_board()（bitboard 游戏在导出前需要同步）
template <typename G>
concept HasSyncToBoard = requires(G g) { g.sync_to_board(); };

// ── Board I/O 辅助（Traits 游戏）─────────────────────────────────────────

/// 将棋盘导出为 float32 ndarray（Traits 游戏通用）
template <typename Game>
    requires requires { Game::Traits::ROWS; Game::Traits::COLS; }
py::array_t<float> board_to_numpy(Game &self)
{
    if constexpr (HasSyncToBoard<Game>)
        self.sync_to_board();

    constexpr int R = Game::Traits::ROWS;
    constexpr int C = Game::Traits::COLS;
    py::array_t<float> arr({R, C});
    auto buf = arr.mutable_unchecked<2>();
    const int8_t *data = self.board_data();
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            buf(r, c) = static_cast<float>(data[r * C + c]);
    return arr;
}

/// 从 float32 ndarray 导入棋盘，重建内部状态，推算 turn（Traits 游戏通用）
template <typename Game>
    requires requires { Game::Traits::ROWS; Game::Traits::COLS; }
void numpy_to_board(Game &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    constexpr int R = Game::Traits::ROWS;
    constexpr int C = Game::Traits::COLS;
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != R || buf.shape(1) != C)
        throw std::runtime_error("board shape must be (" + std::to_string(R) +
                                 ", " + std::to_string(C) + ")");

    int8_t tmp[R * C];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            tmp[r * C + c] = static_cast<int8_t>(buf(r, c));
    self.import_board(tmp);
    self.set_turn(self.n_pieces % 2 == 0 ? 1 : -1);
}

/// 从 ndarray 工厂构造游戏实例（Traits 游戏通用）
template <typename Game>
    requires requires { Game::Traits::ROWS; Game::Traits::COLS; }
Game construct_from_board(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    Game g;
    numpy_to_board(g, arr);
    return g;
}

// ── NN 输入张量（所有游戏通用）──────────────────────────────────────────

/**
 * 构建 (1, 3, rows, cols) float32 NN 输入张量。
 *   通道 0 = 当前落子方棋子位置
 *   通道 1 = 对手棋子位置
 *   通道 2 = 当前落子方标识（全 1 或全 -1）
 *
 * 使用 board_data() + get_turn()，所有游戏均支持。
 * 对有 sync_to_board() 的游戏自动调用同步。
 */
template <typename Game>
py::array_t<float> make_current_state(Game &self, int rows, int cols)
{
    if constexpr (HasSyncToBoard<Game>)
        self.sync_to_board();

    py::array_t<float> state({1, 3, rows, cols});
    auto buf = state.mutable_unchecked<4>();
    std::memset(buf.mutable_data(0, 0, 0, 0), 0,
                sizeof(float) * static_cast<size_t>(3 * rows * cols));

    const int turn = self.get_turn();
    const float turn_f = static_cast<float>(turn);
    const int8_t *data = self.board_data();

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
        {
            const int8_t v = data[r * cols + c];
            if (v == turn)
                buf(0, 0, r, c) = 1.0f;
            else if (v == -turn)
                buf(0, 1, r, c) = 1.0f;
            buf(0, 2, r, c) = turn_f;
        }
    return state;
}

// ── 公共绑定注册（所有游戏共享）─────────────────────────────────────────

/**
 * 注册所有游戏环境共享的 pybind11 方法。
 *
 * 包括：reset, copy, step, winPlayer, check_winner, check_full,
 *       turn 属性, NUM_SYMMETRIES 静态属性,
 *       apply_symmetry（已修复 const_cast UB）, random_symmetry
 *
 * @param cls          py::class_<Game> 引用
 * @param num_sym      对称变换数量
 */
template <typename Game>
void register_env_common(py::class_<Game> &cls, int num_sym)
{
    cls
        .def("reset", &Game::reset)

        .def("copy", [](const Game &self)
             { return Game(self); })

        .def("step", &Game::step, py::arg("action"))

        .def("winPlayer", &Game::check_winner)
        .def("check_winner", &Game::check_winner)
        .def("check_full", &Game::is_full)

        .def_property(
            "turn",
            &Game::get_turn,
            &Game::set_turn)

        .def_property_readonly_static(
            "NUM_SYMMETRIES",
            [num_sym](py::object) { return num_sym; })

        // apply_symmetry: 接收非 const 引用，避免 const_cast UB
        .def(
            "apply_symmetry", [](Game &self, int sym_id, bool inplace) -> Game
            {
                if (inplace)
                {
                    self.apply_symmetry(sym_id);
                    return self;
                }
                Game copy(self);
                copy.apply_symmetry(sym_id);
                return copy; },
            py::arg("sym_id"), py::arg("inplace") = false)

        .def(
            "random_symmetry", [num_sym](const Game &self) -> py::tuple
            {
                static thread_local std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, num_sym - 1);
                int sym_id = dist(rng);
                Game copy(self);
                copy.apply_symmetry(sym_id);
                return py::make_tuple(copy, sym_id); });
}

// ── Traits 游戏额外公共绑定 ─────────────────────────────────────────────

/**
 * 注册 Traits 游戏（编译期维度）的额外公共方法。
 *
 * 包括：board 属性, valid_move, valid_mask, current_state, pickle
 *
 * @param cls  py::class_<Game> 引用
 */
template <typename Game>
    requires requires { Game::Traits::ROWS; Game::Traits::COLS; Game::Traits::ACTION_SIZE; }
void register_env_traits(py::class_<Game> &cls)
{
    constexpr int R = Game::Traits::ROWS;
    constexpr int C = Game::Traits::COLS;
    constexpr int A = Game::Traits::ACTION_SIZE;

    cls
        // board 属性
        .def_property(
            "board",
            &board_to_numpy<Game>,
            &numpy_to_board<Game>)

        // 合法动作列表
        .def(
            "valid_move", [](const Game &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                for (int i = 0; i < vm.count; ++i)
                    result.append(vm.moves[i]);
                return result; })

        // 合法动作掩码
        .def(
            "valid_mask", [](const Game &self)
            {
                auto vm = self.get_valid_moves();
                py::list result;
                bool mask[A] = {};
                for (int i = 0; i < vm.count; ++i)
                    mask[vm.moves[i]] = true;
                for (int a = 0; a < A; ++a)
                    result.append(mask[a]);
                return result; })

        // NN 输入张量 (1, 3, R, C)
        .def(
            "current_state", [](Game &self)
            { return make_current_state(self, R, C); })

        // Pickle: (board_ndarray, turn)
        .def(py::pickle(
            [](Game &self) -> py::tuple
            {
                return py::make_tuple(board_to_numpy(self), self.get_turn());
            },
            [](py::tuple t) -> Game
            {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid pickle state");
                Game g;
                numpy_to_board(g, t[0].cast<py::array_t<float>>());
                g.set_turn(t[1].cast<int>());
                return g;
            }));
}

} // namespace env_bind
