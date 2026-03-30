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
        .def(py::init<int, int>(), py::arg("max_seq_len") = 1024, py::arg("kv_bits") = 0,
             "Create inference engine. kv_bits=0: fp16 KV cache, kv_bits=2: TurboQuant 2-bit (8x compression)")

        .def("kv_bits", &InferenceEngine::kv_bits,
             "KV cache quantization bits (0=fp16, 2=TurboQuant 2-bit)")

        .def("load_weights", &InferenceEngine::load_weights,
             py::arg("model_dir"),
             "Load model weights from safetensors directory")

        .def("load_lora", &InferenceEngine::load_lora,
             py::arg("lora_dir"), py::arg("scale") = 1.0f,
             "Load LoRA adapter weights")

        .def("reset", &InferenceEngine::reset,
             "Reset KV cache for new generation")

        .def("share_embedding", [](InferenceEngine& self, size_t ptr) {
                 self.share_embedding(reinterpret_cast<void*>(ptr));
             },
             py::arg("data_ptr"),
             "Share embedding from PyTorch tensor (pass tensor.data_ptr())")

        .def("cache_weights", &InferenceEngine::cache_weights,
             "Pre-dequant Q4L weights to fp16 for fast batched GEMM")

        .def("set_arena", [](InferenceEngine& self, size_t ptr, size_t size) {
                 self.set_arena(reinterpret_cast<void*>(ptr), size);
             },
             py::arg("data_ptr"), py::arg("size"),
             "Set external arena memory (pass torch tensor.data_ptr() and nbytes)")

        .def("sleep", &InferenceEngine::sleep,
             "Free GPU buffers (call before PyTorch training phase)")

        .def("wake", &InferenceEngine::wake,
             "Re-allocate GPU buffers (call before generation phase)")

        .def("generate", &InferenceEngine::generate,
             py::arg("prompt"),
             py::arg("max_new_tokens") = 512,
             py::arg("temperature") = 1.0f,
             py::arg("top_p") = 0.9f,
             py::arg("eos_token_id") = -1,
             py::arg("stop_token_ids") = std::vector<int>{},
             "Generate tokens from prompt. Returns list of generated token IDs.")

        .def("sample_greedy_gpu", &InferenceEngine::sample_greedy_gpu,
             "Greedy sample on GPU (4 bytes copy instead of 600KB)")

        .def("sample_gpu", &InferenceEngine::sample_gpu,
             py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
             "GPU sampling with temperature + top-p (4 bytes copy)")

        .def("update_lora", [](InferenceEngine& self,
                                int layer_idx, const std::string& proj_name,
                                py::buffer A_buf, py::buffer B_buf, float scale) {
                 auto A_info = A_buf.request();
                 auto B_info = B_buf.request();
                 self.update_lora_weight(
                     layer_idx, proj_name.c_str(),
                     static_cast<const half*>(A_info.ptr), A_info.shape[0], A_info.shape[1],
                     static_cast<const half*>(B_info.ptr), B_info.shape[0], B_info.shape[1],
                     scale);
             },
             py::arg("layer_idx"), py::arg("proj_name"),
             py::arg("A"), py::arg("B"), py::arg("scale") = 1.0f,
             "Update a single LoRA adapter (pass numpy fp16 arrays)")

        .def("update_lora_gpu", [](InferenceEngine& self,
                                    int layer_idx, const std::string& proj_name,
                                    size_t A_ptr, int A_rows, int A_cols,
                                    size_t B_ptr, int B_rows, int B_cols, float scale) {
                 self.update_lora_weight(
                     layer_idx, proj_name.c_str(),
                     reinterpret_cast<const half*>(A_ptr), A_rows, A_cols,
                     reinterpret_cast<const half*>(B_ptr), B_rows, B_cols,
                     scale);
             },
             py::arg("layer_idx"), py::arg("proj_name"),
             py::arg("A_ptr"), py::arg("A_rows"), py::arg("A_cols"),
             py::arg("B_ptr"), py::arg("B_rows"), py::arg("B_cols"),
             py::arg("scale") = 1.0f,
             "Update LoRA from GPU pointers (pass tensor.data_ptr(), avoids CPU roundtrip)")

        .def("decode_token", &InferenceEngine::decode,
             py::arg("token_id"),
             "Process one token through the model")

        .def("prefill", [](InferenceEngine& self, const std::vector<int>& tokens) {
                self.prefill(tokens.data(), tokens.size());
             },
             py::arg("token_ids"),
             "Process prompt tokens (prefill phase)")

        .def("generate_batch", &InferenceEngine::generate_batch,
             py::arg("prompts"),
             py::arg("max_new_tokens") = 512,
             py::arg("temperature") = 1.0f,
             py::arg("top_p") = 0.9f,
             py::arg("eos_token_id") = -1,
             py::arg("stop_token_ids") = std::vector<int>{},
             "Generate from G prompts in parallel (GEMM, tensor cores).")

        .def("model_config", [](InferenceEngine& self) {
                 const auto& c = self.config();
                 py::dict d;
                 d["hidden_size"] = c.hidden_size;
                 d["intermediate_size"] = c.intermediate_size;
                 d["num_layers"] = c.num_layers;
                 d["num_heads"] = c.num_heads;
                 d["num_kv_heads"] = c.num_kv_heads;
                 d["head_dim"] = c.head_dim;
                 d["vocab_size"] = c.vocab_size;
                 return d;
             },
             "Get model configuration (loaded from config.json alongside weights)");
}
