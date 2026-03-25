/**
 * Python bindings for the Jetson LLM inference engine.
 *
 * Usage from Python:
 *   import jetson_engine
 *   engine = jetson_engine.Engine(max_seq_len=1024)
 *   engine.load_weights("/path/to/model")
 *   engine.load_lora("/path/to/lora", scale=1.0)
 *   tokens = engine.generate(prompt_ids, max_tokens=512, temperature=1.0)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "model.h"

namespace py = pybind11;

PYBIND11_MODULE(jetson_engine, m) {
    m.doc() = "Jetson LLM Inference Engine — fast generation for small transformers";

    py::class_<InferenceEngine>(m, "Engine")
        .def(py::init<int>(), py::arg("max_seq_len") = 1024,
             "Create inference engine with pre-allocated KV cache")

        .def("load_weights", &InferenceEngine::load_weights,
             py::arg("model_dir"),
             "Load model weights from safetensors directory")

        .def("load_lora", &InferenceEngine::load_lora,
             py::arg("lora_dir"), py::arg("scale") = 1.0f,
             "Load LoRA adapter weights")

        .def("reset", &InferenceEngine::reset,
             "Reset KV cache for new generation")

        .def("generate", &InferenceEngine::generate,
             py::arg("prompt"),
             py::arg("max_new_tokens") = 512,
             py::arg("temperature") = 1.0f,
             py::arg("top_p") = 0.9f,
             py::arg("eos_token_id") = -1,
             py::arg("stop_token_ids") = std::vector<int>{},
             "Generate tokens from prompt. Returns list of generated token IDs.")

        .def("sample", &InferenceEngine::sample,
             py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
             "Sample from current logits (CPU-side, full top-p)")

        .def("sample_greedy_gpu", &InferenceEngine::sample_greedy_gpu,
             "Greedy sample on GPU (4 bytes copy instead of 600KB)")

        .def("decode_token", &InferenceEngine::decode,
             py::arg("token_id"),
             "Process one token through the model")

        .def("prefill", [](InferenceEngine& self, const std::vector<int>& tokens) {
                self.prefill(tokens.data(), tokens.size());
             },
             py::arg("token_ids"),
             "Process prompt tokens (prefill phase)");
}
