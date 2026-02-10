#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BatchedMCTS.h"

namespace py = pybind11;
using namespace AlphaZero;

PYBIND11_MODULE(mcts_cpp, m)
{
    m.doc() = "AlphaZero Batched MCTS plugin using pybind11";

    py::class_<BatchedMCTS>(m, "BatchedMCTS")
        .def(py::init<int, float, float, float, float>(),
             py::arg("n_envs"), py::arg("c_init"), py::arg("c_base"),
             py::arg("discount"), py::arg("alpha"))

        .def("set_seed", &BatchedMCTS::set_seed, "Set random seed for all OpenMP threads")

        .def("reset_env", &BatchedMCTS::reset_env)

        .def("get_all_counts", &BatchedMCTS::get_all_counts)

        .def("prune_roots", [](BatchedMCTS &self, py::array_t<int> actions)
             {
            py::buffer_info buf = actions.request();

            if (buf.ndim != 1) throw std::runtime_error("Actions must be 1D array");

            std::span<const int> s(static_cast<int*>(buf.ptr), buf.size);
            self.prune_roots(s); })

        .def("search_batch", [](BatchedMCTS &self, py::array_t<int8_t> input_boards, py::array_t<int> turns)
             {
            auto buf_in = input_boards.request();
            auto buf_turns = turns.request();
            int batch_size = buf_in.shape[0];
            
            py::array_t<int8_t> out_boards({batch_size, 6, 7});

            py::array_t<float> out_vals(batch_size);
            py::array_t<uint8_t> out_term(batch_size);

            int8_t* ptr_in = static_cast<int8_t*>(buf_in.ptr);
            int* ptr_turns = static_cast<int*>(buf_turns.ptr);

            int8_t* ptr_out_boards = static_cast<int8_t*>(out_boards.request().ptr);
            float* ptr_out_vals = static_cast<float*>(out_vals.request().ptr);
            uint8_t* ptr_out_term = static_cast<uint8_t*>(out_term.request().ptr);
            {
                py::gil_scoped_release release; // <--- 关键！释放 Python 锁
                self.search_batch(ptr_in, ptr_turns, ptr_out_boards, ptr_out_vals, ptr_out_term);
            }
            return py::make_tuple(out_boards, out_vals, out_term); })

        .def("backprop_batch", [](BatchedMCTS &self,
                                  py::array_t<float> policy_logits,
                                  py::array_t<float> values,
                                  py::array_t<uint8_t> is_term)
             {
            
            auto buf_pol = policy_logits.request();
            auto buf_val = values.request();
            auto buf_term = is_term.request();

            float* ptr_pol = static_cast<float*>(buf_pol.ptr);
            float* ptr_val = static_cast<float*>(buf_val.ptr);
            uint8_t* ptr_term = static_cast<uint8_t*>(buf_term.ptr);

            {
                py::gil_scoped_release release; // 释放 GIL
                self.backprop_batch(ptr_pol, ptr_val, ptr_term);
            } });
}