/**
 * MCTS pybind11 绑定 — 将 BatchedMCTS<Game> 暴露为 Python mcts_cpp 模块。
 *
 * 通过模板函数 register_batched_mcts<Game>() 实现多游戏支持：
 * 添加新游戏只需 #include 对应头文件并调用 register_batched_mcts<NewGame>()。
 *
 * 所有 C++ 计算阶段（search_batch, backprop_batch, rollout_playout）
 * 在执行前释放 GIL，允许 Python 线程并行运行。
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BatchedMCTS.h"
#include "Connect4.h"
// 添加新游戏时在这里 include，例如:
// #include "TicTacToe.h"

namespace py = pybind11;
using namespace AlphaZero;

/**
 * 通用注册函数：将 BatchedMCTS<Game> 绑定到 Python。
 * 所有维度（ACTION_SIZE, BOARD_SIZE, BOARD_SHAPE）自动从 Game::Traits 获取。
 *
 * @tparam Game 满足 MCTSGame concept 的游戏类型
 * @param m     Python 模块
 * @param name  Python 类名（如 "BatchedMCTS_Connect4"）
 */
template <MCTSGame Game>
void register_batched_mcts(py::module_ &m, const char *name)
{
    using BM = BatchedMCTS<Game>;
    constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
    constexpr int BOARD_SIZE = Game::Traits::BOARD_SIZE;

    py::class_<BM>(m, name)
        // ── 构造 ─────────────────────────────────────────────────────────
        .def(py::init<int, float, float, float, float, float, bool, float, float>(),
             py::arg("n_envs"), py::arg("c_init"), py::arg("c_base"),
             py::arg("alpha"), py::arg("noise_epsilon") = 0.25f,
             py::arg("fpu_reduction") = 0.4f, py::arg("use_symmetry") = true,
             py::arg("mlh_slope") = 0.0f, py::arg("mlh_cap") = 0.2f)

        // ── 参数设置 ─────────────────────────────────────────────────────
        .def("set_seed", &BM::set_seed,
             "Set random seed for all OpenMP threads")

        .def("set_noise_epsilon", &BM::set_noise_epsilon,
             py::arg("eps"),
             "Set Dirichlet noise mixing weight for all environments")

        .def("set_mlh_params", &BM::set_mlh_params,
             py::arg("slope"), py::arg("cap"),
             "Set Moves Left Head parameters (slope and cap)")

        // ── 树管理 ──────────────────────────────────────────────────────
        .def("reset_env", &BM::reset_env,
             "Reset MCTS tree for specified environment index")

        .def("get_num_envs", &BM::get_num_envs,
             "Return number of parallel environments")

        /// 批量树剪枝：将各环境选中动作的子树提升为新根节点
        .def("prune_roots", [](BM &self,
                               py::array_t<int, py::array::c_style | py::array::forcecast> actions)
             {
            py::buffer_info buf = actions.request();
            if (buf.ndim != 1) throw std::runtime_error("Actions must be 1D array");
            if (buf.size != self.get_num_envs())
                throw std::runtime_error("prune_roots: actions size (" + std::to_string(buf.size) +
                                         ") must match n_envs (" + std::to_string(self.get_num_envs()) + ")");
            std::span<const int> s(static_cast<int*>(buf.ptr), buf.size);
            self.prune_roots(s); })

        // ── AlphaZero MCTS（NN 评估）──────────────────────────────────

        /**
         * Selection 阶段：并行选择叶节点。
         * 输入当前棋盘状态，返回 (leaf_boards, term_values, is_terminal, leaf_turns)。
         * Python 端用返回的叶节点状态调用 NN，再调 backprop_batch() 反向传播。
         */
        .def("search_batch", [](BM &self,
                                py::array_t<int8_t, py::array::c_style | py::array::forcecast> input_boards,
                                py::array_t<int, py::array::c_style | py::array::forcecast> turns)
             {
            auto buf_in = input_boards.request();
            auto buf_turns = turns.request();
            py::ssize_t batch_size = buf_in.shape[0];

            if (batch_size != self.get_num_envs())
                throw std::runtime_error("search_batch: input_boards batch size (" + std::to_string(batch_size) +
                                         ") must match n_envs (" + std::to_string(self.get_num_envs()) + ")");
            if (buf_turns.size != batch_size) throw std::runtime_error("Turns size must match batch size");

            // 从 Traits 动态构建输出形状
            std::vector<py::ssize_t> out_shape = {batch_size};
            for (auto d : Game::Traits::BOARD_SHAPE) out_shape.push_back(d);

            py::array_t<int8_t> out_boards(out_shape);
            py::array_t<float> out_vals(batch_size);
            py::array_t<uint8_t> out_term(batch_size);
            py::array_t<int> out_turns(batch_size);

            int8_t* ptr_in = static_cast<int8_t*>(buf_in.ptr);
            int* ptr_turns = static_cast<int*>(buf_turns.ptr);

            int8_t* ptr_out_boards = static_cast<int8_t*>(out_boards.request().ptr);
            float* ptr_out_vals = static_cast<float*>(out_vals.request().ptr);
            uint8_t* ptr_out_term = static_cast<uint8_t*>(out_term.request().ptr);
            int* ptr_out_turns = static_cast<int*>(out_turns.request().ptr);

            {
                py::gil_scoped_release release;
                self.search_batch(ptr_in, ptr_turns, ptr_out_boards, ptr_out_vals, ptr_out_term, ptr_out_turns);
            }

            return py::make_tuple(out_boards, out_vals, out_term, out_turns); })

        /**
         * Backpropagation 阶段：用 NN 评估结果展开叶节点并反向传播。
         * 接收 policy_logits, values, moves_left, is_terminal 四个批量数组。
         */
        .def("backprop_batch", [](BM &self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> policy_logits,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> values,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> moves_left,
                                  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_term)
             {
            auto buf_pol = policy_logits.request();
            auto buf_val = values.request();
            auto buf_ml = moves_left.request();
            auto buf_term = is_term.request();

            int n = self.get_num_envs();
            if (buf_pol.shape[0] != n)
                throw std::runtime_error("backprop_batch: policy_logits batch size (" + std::to_string(buf_pol.shape[0]) +
                                         ") must match n_envs (" + std::to_string(n) + ")");
            if (buf_val.size != n)
                throw std::runtime_error("backprop_batch: values size (" + std::to_string(buf_val.size) +
                                         ") must match n_envs (" + std::to_string(n) + ")");
            if (buf_ml.size != n)
                throw std::runtime_error("backprop_batch: moves_left size (" + std::to_string(buf_ml.size) +
                                         ") must match n_envs (" + std::to_string(n) + ")");
            if (buf_term.size != n)
                throw std::runtime_error("backprop_batch: is_term size (" + std::to_string(buf_term.size) +
                                         ") must match n_envs (" + std::to_string(n) + ")");

            float* ptr_pol = static_cast<float*>(buf_pol.ptr);
            float* ptr_val = static_cast<float*>(buf_val.ptr);
            float* ptr_ml = static_cast<float*>(buf_ml.ptr);
            uint8_t* ptr_term = static_cast<uint8_t*>(buf_term.ptr);

            {
                py::gil_scoped_release release;
                self.backprop_batch(ptr_pol, ptr_val, ptr_ml, ptr_term);
            } })

        // ── 纯 MCTS（Random Rollout）─────────────────────────────────

        /**
         * 纯 MCTS playout：整个 n_playout 循环在 C++ 内完成。
         * 叶节点用 uniform prior 展开，价值由随机模拟到终局得到，无需 Python 回调。
         */
        .def("rollout_playout", [](BM &self,
                                   py::array_t<int8_t, py::array::c_style | py::array::forcecast> input_boards,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> turns,
                                   int n_playout)
             {
            auto buf_in = input_boards.request();
            auto buf_turns = turns.request();
            py::ssize_t batch_size = buf_in.shape[0];

            if (batch_size != self.get_num_envs())
                throw std::runtime_error("rollout_playout: input_boards batch size (" + std::to_string(batch_size) +
                                         ") must match n_envs (" + std::to_string(self.get_num_envs()) + ")");
            if (buf_turns.size != batch_size)
                throw std::runtime_error("rollout_playout: turns size must match batch size");

            int8_t* ptr_in = static_cast<int8_t*>(buf_in.ptr);
            int* ptr_turns = static_cast<int*>(buf_turns.ptr);

            {
                py::gil_scoped_release release;
                self.rollout_playout(ptr_in, ptr_turns, n_playout);
            } },
             py::arg("input_boards"), py::arg("turns"), py::arg("n_playout"),
             "Run pure MCTS with random rollout evaluation entirely in C++")

        // ── 统计查询 ─────────────────────────────────────────────────────

        /// 获取所有环境根节点各动作的访问次数 (flat: n_envs × ACTION_SIZE)
        .def("get_all_counts", &BM::get_all_counts)

        /**
         * 获取所有环境的根节点统计量。
         * 返回 shape (n_envs, 3 + action_size*5) 的 float32 数组。
         * 每行布局：[root_N, root_Q, root_M, a0_N, a0_Q, a0_prior, a0_noise, a0_M, ...]
         */
        .def("get_all_root_stats", [](BM &self)
             {
            int n = self.get_num_envs();
            constexpr int S = BM::STATS_PER_ENV;
            py::array_t<float> out({n, S});
            float* ptr = static_cast<float*>(out.request().ptr);
            self.get_all_root_stats(ptr);
            return out; },
             "Returns root node stats: shape (n_envs, 3 + action_size*5)")

        // ── 游戏维度常量（class-level 只读属性）──────────────────────────
        .def_property_readonly_static("action_size",
            [](py::object) { return ACTION_SIZE; })
        .def_property_readonly_static("board_size",
            [](py::object) { return BOARD_SIZE; })
        .def_property_readonly_static("board_shape",
            [](py::object) {
                py::tuple shape(Game::Traits::BOARD_SHAPE.size());
                for (size_t i = 0; i < Game::Traits::BOARD_SHAPE.size(); ++i)
                    shape[i] = Game::Traits::BOARD_SHAPE[i];
                return shape;
            });
}

PYBIND11_MODULE(mcts_cpp, m)
{
    m.doc() = "AlphaZero Batched MCTS (multi-game support)";

    // === 注册各游戏的 BatchedMCTS ===
    register_batched_mcts<Connect4>(m, "BatchedMCTS_Connect4");

    // 添加新游戏只需一行:
    // register_batched_mcts<TicTacToe>(m, "BatchedMCTS_TicTacToe");
    // register_batched_mcts<Gomoku>(m, "BatchedMCTS_Gomoku");
}
