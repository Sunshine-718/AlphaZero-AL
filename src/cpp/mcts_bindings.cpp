/**
 * MCTS pybind11 绑定 — 将 BatchedMCTS<Game> 暴露为 Python mcts_cpp 模块。
 *
 * 通过模板函数 register_batched_mcts<Game>() 实现多游戏支持：
 * 添加新游戏只需 #include 对应头文件并调用 register_batched_mcts<NewGame>()。
 *
 * 所有 C++ 计算阶段（search_batch, backprop_batch, search）
 * 在执行前释放 GIL，允许 Python 线程并行运行。
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BatchedMCTS.h"
#include "IEvaluator.h"
#include "RolloutEvaluator.h"
#include "Connect4.h"
#include "Othello.h"

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
    using IE = IEvaluator<Game>;
    using RE = RolloutEvaluator<Game>;
    constexpr int ACTION_SIZE = Game::Traits::ACTION_SIZE;
    constexpr int BOARD_SIZE = Game::Traits::BOARD_SIZE;

    // ── 注册 IEvaluator 基类和 RolloutEvaluator 子类 ──────────────
    std::string ie_name = std::string(name);
    ie_name.replace(0, 11, "IEvaluator");  // "BatchedMCTS_X" → "IEvaluator_X"
    std::string re_name = std::string(name);
    re_name.replace(0, 11, "RolloutEvaluator");  // "BatchedMCTS_X" → "RolloutEvaluator_X"

    py::class_<IE>(m, ie_name.c_str());
    py::class_<RE, IE>(m, re_name.c_str())
        .def(py::init<>());

    py::class_<BM>(m, name)
        // ── 构造 ─────────────────────────────────────────────────────────
        .def(py::init<int>(), py::arg("n_envs"))

        // ── 配置访问 ────────────────────────────────────────────────────
        .def_property("config",
            [](BM &self) -> SearchConfig& { return self.config(); },
            [](BM &self, const SearchConfig &cfg) { self.config() = cfg; },
            py::return_value_policy::reference_internal)

        // ── 种子设置 ────────────────────────────────────────────────────
        .def("set_seed", &BM::set_seed,
             "Set random seed for all OpenMP threads")

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
         * 输入当前棋盘状态，返回 (leaf_boards, term_d, term_p1w, term_p2w, is_terminal, leaf_turns)。
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
            py::array_t<float> out_term_d(batch_size);
            py::array_t<float> out_term_p1w(batch_size);
            py::array_t<float> out_term_p2w(batch_size);
            py::array_t<uint8_t> out_term(batch_size);
            py::array_t<int> out_turns(batch_size);

            int8_t* ptr_in = static_cast<int8_t*>(buf_in.ptr);
            int* ptr_turns = static_cast<int*>(buf_turns.ptr);

            int8_t* ptr_out_boards = static_cast<int8_t*>(out_boards.request().ptr);
            float* ptr_out_td = static_cast<float*>(out_term_d.request().ptr);
            float* ptr_out_tp1w = static_cast<float*>(out_term_p1w.request().ptr);
            float* ptr_out_tp2w = static_cast<float*>(out_term_p2w.request().ptr);
            uint8_t* ptr_out_term = static_cast<uint8_t*>(out_term.request().ptr);
            int* ptr_out_turns = static_cast<int*>(out_turns.request().ptr);

            {
                py::gil_scoped_release release;
                self.search_batch(ptr_in, ptr_turns, ptr_out_boards,
                                  ptr_out_td, ptr_out_tp1w, ptr_out_tp2w,
                                  ptr_out_term, ptr_out_turns);
            }

            return py::make_tuple(out_boards, out_term_d, out_term_p1w, out_term_p2w, out_term, out_turns); })

        /**
         * Backpropagation 阶段：用 NN 评估结果展开叶节点并反向传播。
         */
        .def("backprop_batch", [](BM &self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> policy_logits,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> d_vals,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> p1w_vals,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> p2w_vals,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> moves_left,
                                  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_term)
             {
            auto buf_pol = policy_logits.request();
            auto buf_d = d_vals.request();
            auto buf_p1w = p1w_vals.request();
            auto buf_p2w = p2w_vals.request();
            auto buf_ml = moves_left.request();
            auto buf_term = is_term.request();

            int n = self.get_num_envs();
            if (buf_pol.shape[0] != n)
                throw std::runtime_error("backprop_batch: policy_logits batch size (" + std::to_string(buf_pol.shape[0]) +
                                         ") must match n_envs (" + std::to_string(n) + ")");
            if (buf_d.size != n || buf_p1w.size != n || buf_p2w.size != n)
                throw std::runtime_error("backprop_batch: d/p1w/p2w size must match n_envs (" + std::to_string(n) + ")");
            if (buf_ml.size != n)
                throw std::runtime_error("backprop_batch: moves_left size (" + std::to_string(buf_ml.size) +
                                         ") must match n_envs (" + std::to_string(n) + ")");
            if (buf_term.size != n)
                throw std::runtime_error("backprop_batch: is_term size (" + std::to_string(buf_term.size) +
                                         ") must match n_envs (" + std::to_string(n) + ")");

            float* ptr_pol = static_cast<float*>(buf_pol.ptr);
            float* ptr_d = static_cast<float*>(buf_d.ptr);
            float* ptr_p1w = static_cast<float*>(buf_p1w.ptr);
            float* ptr_p2w = static_cast<float*>(buf_p2w.ptr);
            float* ptr_ml = static_cast<float*>(buf_ml.ptr);
            uint8_t* ptr_term = static_cast<uint8_t*>(buf_term.ptr);

            {
                py::gil_scoped_release release;
                self.backprop_batch(ptr_pol, ptr_d, ptr_p1w, ptr_p2w, ptr_ml, ptr_term);
            } })

        // ── Virtual Loss 批量搜索 ────────────────────────────────────

        /// 移除所有环境的 VL (n_inflight)。用于异常安全清理。
        .def("remove_all_vl", [](BM &self, int K)
             {
            {
                py::gil_scoped_release release;
                self.remove_all_vl(K);
            }
        }, py::arg("K"),
           "Remove all VL (n_inflight) from all trees. For exception-safety cleanup.")

        /**
         * VL Selection 阶段：每棵树执行 K 次 VL 模拟，返回 N*K 个叶节点。
         * 返回 (boards, term_d, term_p1w, term_p2w, is_term, turns, sym_ids)。
         */
        .def("search_batch_vl", [](BM &self,
                                   int K,
                                   py::array_t<int8_t, py::array::c_style | py::array::forcecast> input_boards,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> turns)
             {
            auto buf_in = input_boards.request();
            auto buf_turns = turns.request();
            py::ssize_t n = self.get_num_envs();

            if (buf_in.shape[0] != n)
                throw std::runtime_error("search_batch_vl: input batch (" + std::to_string(buf_in.shape[0]) +
                                         ") != n_envs (" + std::to_string(n) + ")");
            if (buf_turns.size != n)
                throw std::runtime_error("search_batch_vl: turns size must match n_envs");
            if (K < 1)
                throw std::runtime_error("search_batch_vl: K must be >= 1");

            py::ssize_t total = n * K;

            // 输出数组：N*K 个叶节点
            std::vector<py::ssize_t> out_shape = {total};
            for (auto d : Game::Traits::BOARD_SHAPE) out_shape.push_back(d);

            py::array_t<int8_t>  out_boards(out_shape);
            py::array_t<float>   out_term_d(total);
            py::array_t<float>   out_term_p1w(total);
            py::array_t<float>   out_term_p2w(total);
            py::array_t<uint8_t> out_term(total);
            py::array_t<int>     out_turns(total);
            py::array_t<int>     out_sym_ids(total);

            int8_t*  ptr_in       = static_cast<int8_t*>(buf_in.ptr);
            int*     ptr_turns_in = static_cast<int*>(buf_turns.ptr);
            int8_t*  ptr_ob  = static_cast<int8_t*>(out_boards.request().ptr);
            float*   ptr_td  = static_cast<float*>(out_term_d.request().ptr);
            float*   ptr_tp1 = static_cast<float*>(out_term_p1w.request().ptr);
            float*   ptr_tp2 = static_cast<float*>(out_term_p2w.request().ptr);
            uint8_t* ptr_ot  = static_cast<uint8_t*>(out_term.request().ptr);
            int*     ptr_otn = static_cast<int*>(out_turns.request().ptr);
            int*     ptr_sym = static_cast<int*>(out_sym_ids.request().ptr);

            {
                py::gil_scoped_release release;
                self.search_batch_vl(K, ptr_in, ptr_turns_in, ptr_ob,
                                     ptr_td, ptr_tp1, ptr_tp2,
                                     ptr_ot, ptr_otn, ptr_sym);
            }

            return py::make_tuple(out_boards, out_term_d, out_term_p1w,
                                  out_term_p2w, out_term, out_turns, out_sym_ids);
        }, py::arg("K"), py::arg("input_boards"), py::arg("turns"),
           "VL Selection: K sims per tree, returns N*K leaves + sym_ids")

        /**
         * VL Backpropagation 阶段：移除 VL 并反向传播 N*K 个结果。
         */
        .def("backprop_batch_vl", [](BM &self,
                                     int K,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> policy_logits,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> d_vals,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> p1w_vals,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> p2w_vals,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> moves_left,
                                     py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_term,
                                     py::array_t<int, py::array::c_style | py::array::forcecast> sym_ids)
             {
            auto buf_pol = policy_logits.request();
            auto buf_d = d_vals.request();
            auto buf_p1w = p1w_vals.request();
            auto buf_p2w = p2w_vals.request();
            auto buf_ml = moves_left.request();
            auto buf_term = is_term.request();
            auto buf_sym = sym_ids.request();

            int n = self.get_num_envs();
            py::ssize_t total = static_cast<py::ssize_t>(n) * K;

            if (buf_pol.shape[0] != total)
                throw std::runtime_error("backprop_batch_vl: policy batch (" + std::to_string(buf_pol.shape[0]) +
                                         ") != N*K (" + std::to_string(total) + ")");
            if (buf_d.size != total || buf_p1w.size != total || buf_p2w.size != total)
                throw std::runtime_error("backprop_batch_vl: d/p1w/p2w size must be N*K");
            if (buf_ml.size != total)
                throw std::runtime_error("backprop_batch_vl: moves_left size must be N*K");
            if (buf_term.size != total)
                throw std::runtime_error("backprop_batch_vl: is_term size must be N*K");
            if (buf_sym.size != total)
                throw std::runtime_error("backprop_batch_vl: sym_ids size must be N*K");

            float*   ptr_pol  = static_cast<float*>(buf_pol.ptr);
            float*   ptr_d    = static_cast<float*>(buf_d.ptr);
            float*   ptr_p1w  = static_cast<float*>(buf_p1w.ptr);
            float*   ptr_p2w  = static_cast<float*>(buf_p2w.ptr);
            float*   ptr_ml   = static_cast<float*>(buf_ml.ptr);
            uint8_t* ptr_term = static_cast<uint8_t*>(buf_term.ptr);
            int*     ptr_sym  = static_cast<int*>(buf_sym.ptr);

            {
                py::gil_scoped_release release;
                self.backprop_batch_vl(K, ptr_pol, ptr_d, ptr_p1w, ptr_p2w,
                                       ptr_ml, ptr_term, ptr_sym);
            }
        }, py::arg("K"), py::arg("policy_logits"), py::arg("d_vals"),
           py::arg("p1w_vals"), py::arg("p2w_vals"), py::arg("moves_left"),
           py::arg("is_term"), py::arg("sym_ids"),
           "VL Backprop: remove VL then backprop N*K results")

        // ── 通用搜索入口（IEvaluator）─────────────────────────────────

        /**
         * 通用搜索：用 C++ IEvaluator 完成整个 playout 循环。
         */
        .def("search", [](BM &self,
                          IEvaluator<Game> &evaluator,
                          py::array_t<int8_t, py::array::c_style | py::array::forcecast> input_boards,
                          py::array_t<int, py::array::c_style | py::array::forcecast> turns,
                          int n_playout)
             {
            auto buf_in = input_boards.request();
            auto buf_turns = turns.request();
            py::ssize_t batch_size = buf_in.shape[0];

            if (batch_size != self.get_num_envs())
                throw std::runtime_error("search: input_boards batch size (" + std::to_string(batch_size) +
                                         ") must match n_envs (" + std::to_string(self.get_num_envs()) + ")");
            if (buf_turns.size != batch_size)
                throw std::runtime_error("search: turns size must match batch size");

            int8_t* ptr_in = static_cast<int8_t*>(buf_in.ptr);
            int* ptr_turns = static_cast<int*>(buf_turns.ptr);

            {
                py::gil_scoped_release release;
                self.search(evaluator, ptr_in, ptr_turns, n_playout);
            } },
             py::arg("evaluator"), py::arg("input_boards"), py::arg("turns"), py::arg("n_playout"),
             "Run MCTS search with a C++ evaluator (e.g., RolloutEvaluator)")

        // ── 统计查询 ─────────────────────────────────────────────────────

        /// 获取所有环境根节点各动作的访问次数 (flat: n_envs × ACTION_SIZE)
        .def("get_all_counts", &BM::get_all_counts)

        /**
         * 获取所有环境的根节点统计量（绝对视角）。
         * 返回 shape (n_envs, 6 + action_size*8) 的 float32 数组。
         */
        .def("get_all_root_stats", [](BM &self)
             {
            int n = self.get_num_envs();
            constexpr int S = BM::STATS_PER_ENV;
            py::array_t<float> out({n, S});
            float* ptr = static_cast<float*>(out.request().ptr);
            self.get_all_root_stats(ptr);
            return out; },
             "Returns root node stats: shape (n_envs, 6 + action_size*8)")

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

    // === 注册 SearchConfig（模块级，所有游戏共享）===
    py::class_<SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("c_init", &SearchConfig::c_init)
        .def_readwrite("c_base", &SearchConfig::c_base)
        .def_readwrite("dirichlet_alpha", &SearchConfig::dirichlet_alpha)
        .def_readwrite("noise_epsilon", &SearchConfig::noise_epsilon)
        .def_readwrite("fpu_reduction", &SearchConfig::fpu_reduction)
        .def_readwrite("mlh_slope", &SearchConfig::mlh_slope)
        .def_readwrite("mlh_cap", &SearchConfig::mlh_cap)
        .def_readwrite("mlh_threshold", &SearchConfig::mlh_threshold)
        .def_readwrite("value_decay", &SearchConfig::value_decay)
        .def_readwrite("use_symmetry", &SearchConfig::use_symmetry)
        .def_readwrite("vl_count", &SearchConfig::vl_count);

    // === 注册各游戏的 BatchedMCTS ===
    register_batched_mcts<Connect4>(m, "BatchedMCTS_Connect4");
    register_batched_mcts<Othello>(m, "BatchedMCTS_Othello");
}
