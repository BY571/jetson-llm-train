/**
 * Load model weights from HuggingFace safetensors format.
 *
 * Safetensors is a simple binary format:
 * - 8 bytes: header size (uint64 LE)
 * - header_size bytes: JSON header (tensor name -> {dtype, shape, data_offsets})
 * - remaining: raw tensor data
 *
 * For NF4 quantized models (unsloth/Qwen3-0.6B-unsloth-bnb-4bit):
 * - Quantized weights are stored as uint8 with ".weight" suffix
 * - Scale factors stored as ".weight.absmax"
 * - Quant state stored as ".weight.quant_state"
 *
 * For LoRA adapters:
 * - "base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight"
 * - "base_model.model.model.layers.N.self_attn.q_proj.lora_B.weight"
 */

#include "model.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>

// Minimal JSON parser for safetensors header
// (just enough to extract tensor metadata)
// TODO: use a proper JSON library for robustness

struct TensorMeta {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_offset_start;
    int64_t data_offset_end;
};

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0);
    std::string data(size, '\0');
    f.read(&data[0], size);
    return data;
}

void InferenceEngine::load_weights(const std::string& model_dir) {
    std::cout << "Loading weights from: " << model_dir << std::endl;
    // TODO: implement safetensors parsing
    // For now, this is a placeholder that shows the expected interface.
    //
    // The implementation needs to:
    // 1. Read model.safetensors (or model-00001-of-*.safetensors for sharded)
    // 2. Parse the JSON header to find tensor names and offsets
    // 3. For NF4 tensors: load uint8 data + absmax scales
    // 4. For fp16 tensors: load directly (embedding, norms)
    // 5. Allocate CUDA memory and copy
    //
    // Key tensor name patterns:
    //   model.embed_tokens.weight                      -> fp16 embedding
    //   model.layers.{i}.self_attn.q_proj.weight       -> NF4
    //   model.layers.{i}.self_attn.q_proj.weight.absmax -> fp16 scales
    //   model.layers.{i}.input_layernorm.weight         -> fp16 norm
    //   model.norm.weight                              -> fp16 final norm

    std::cout << "  WARNING: weight loading not yet implemented" << std::endl;
    std::cout << "  Use load_weights_from_pytorch() for testing" << std::endl;
}

void InferenceEngine::load_lora(const std::string& lora_dir, float scale) {
    std::cout << "Loading LoRA from: " << lora_dir << std::endl;
    // TODO: implement LoRA safetensors loading
    // LoRA tensor names:
    //   base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight -> (rank, hidden)
    //   base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight -> (out_dim, rank)

    std::cout << "  WARNING: LoRA loading not yet implemented" << std::endl;
}
