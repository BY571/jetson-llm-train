/**
 * Load model weights from .bin + .idx files produced by convert_weights.py.
 *
 * Index format (one line per tensor):
 *   name offset nbytes dtype shape
 *   e.g.: "layers.0.mlp.gate_proj.weight 0 1572864 uint8 1572864,1"
 *
 * NF4 layers have multiple associated tensors:
 *   .weight          (uint8, packed NF4 data)
 *   .weight.absmax   (uint8, quantized scales)
 *   .weight.nested_absmax (float32, second-level scales)
 *   .weight.quant_map (float32, 16 entries NF4 lookup)
 *   .weight.nested_quant_map (float32, 256 entries for absmax dequant)
 */

#include "model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <cmath>

// CPU-side float to fp16 conversion (no __float2half_rn on host)
static uint16_t float_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;

    if (exp <= 0) {
        return sign; // flush to zero
    } else if (exp >= 31) {
        return sign | 0x7C00; // infinity
    }
    return sign | (exp << 10) | (mant >> 13);
}

using namespace qwen3;

struct TensorInfo {
    std::string name;
    size_t offset;
    size_t nbytes;
    std::string dtype;
    std::vector<int> shape;
};

static std::unordered_map<std::string, TensorInfo> load_index(const std::string& idx_path) {
    std::unordered_map<std::string, TensorInfo> index;
    std::ifstream f(idx_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open index: " + idx_path);
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);

        TensorInfo info;
        std::string shape_str;
        ss >> info.name >> info.offset >> info.nbytes >> info.dtype >> shape_str;

        // Parse shape "1572864,1" -> {1572864, 1}
        std::istringstream shape_ss(shape_str);
        std::string dim;
        while (std::getline(shape_ss, dim, ',')) {
            info.shape.push_back(std::stoi(dim));
        }

        index[info.name] = info;
    }
    return index;
}

// Allocate GPU memory and copy raw bytes
static void* gpu_alloc_copy(const char* data, size_t nbytes) {
    void* ptr;
    cudaMalloc(&ptr, nbytes);
    cudaMemcpy(ptr, data, nbytes, cudaMemcpyHostToDevice);
    return ptr;
}

void InferenceEngine::load_weights(const std::string& prefix) {
    std::string idx_path = prefix + ".idx";
    std::string bin_path = prefix + ".bin";

    std::cout << "Loading weights: " << prefix << std::endl;

    // Load index
    auto index = load_index(idx_path);
    std::cout << "  Index: " << index.size() << " tensors" << std::endl;

    // Memory-map the binary file
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open: " + bin_path);
    }
    size_t file_size = f.tellg();
    f.seekg(0);
    std::vector<char> data(file_size);
    f.read(data.data(), file_size);
    f.close();
    std::cout << "  Binary: " << file_size / 1e6 << "MB" << std::endl;

    const char* base = data.data();

    // Helper: load a tensor to GPU, return typed pointer
    auto load_half = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) {
            std::cerr << "  WARN: missing " << name << std::endl;
            return nullptr;
        }
        return (half*)gpu_alloc_copy(base + it->second.offset, it->second.nbytes);
    };

    auto load_uint8 = [&](const std::string& name) -> uint8_t* {
        auto it = index.find(name);
        if (it == index.end()) return nullptr;
        return (uint8_t*)gpu_alloc_copy(base + it->second.offset, it->second.nbytes);
    };

    auto load_float = [&](const std::string& name) -> float* {
        auto it = index.find(name);
        if (it == index.end()) return nullptr;
        return (float*)gpu_alloc_copy(base + it->second.offset, it->second.nbytes);
    };

    auto has = [&](const std::string& name) -> bool {
        return index.find(name) != index.end();
    };

    auto shape0 = [&](const std::string& name) -> int {
        auto it = index.find(name);
        if (it == index.end()) return 0;
        return it->second.shape.empty() ? 0 : it->second.shape[0];
    };

    // Embedding (bf16 or fp16)
    weights_.embed_tokens = load_half("embed_tokens.weight");

    // Final norm
    weights_.final_layernorm = load_half("norm.weight");

    int loaded_layers = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& layer = weights_.layers[i];
        std::string p = "layers." + std::to_string(i) + ".";

        // Attention weights — fp16/bf16 (NOT quantized in unsloth model)
        layer.q_proj_fp16 = load_half(p + "self_attn.q_proj.weight");
        layer.k_proj_fp16 = load_half(p + "self_attn.k_proj.weight");
        layer.v_proj_fp16 = load_half(p + "self_attn.v_proj.weight");
        layer.o_proj_fp16 = load_half(p + "self_attn.o_proj.weight");

        // MLP weights — NF4 quantized
        // For the engine, we need the NF4 data + dequantization info.
        // For now, we dequantize on CPU at load time and store as fp16.
        // TODO: store as NF4 and use nf4_gemv_kernel for memory savings.
        std::string mlp_names[] = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
        half** mlp_ptrs[] = {&layer.gate_proj_fp16, &layer.up_proj_fp16, &layer.down_proj_fp16};
        int mlp_out_dims[] = {INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE};
        int mlp_in_dims[] = {HIDDEN_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE};

        for (int m = 0; m < 3; m++) {
            std::string wname = p + mlp_names[m] + ".weight";
            if (has(wname) && index[wname].dtype == "uint8") {
                // NF4 quantized — dequantize on CPU
                std::string absmax_name = wname + ".absmax";
                std::string nested_absmax_name = wname + ".nested_absmax";
                std::string nested_qmap_name = wname + ".nested_quant_map";
                std::string qmap_name = wname + ".quant_map";

                auto& wi = index[wname];
                auto& ai = index[absmax_name];
                auto& nai = index[nested_absmax_name];
                auto& nqi = index[nested_qmap_name];
                auto& qi = index[qmap_name];

                const uint8_t* packed = (const uint8_t*)(base + wi.offset);
                const uint8_t* absmax_u8 = (const uint8_t*)(base + ai.offset);
                const float* nested_absmax = (const float*)(base + nai.offset);
                const float* nested_qmap = (const float*)(base + nqi.offset);
                const float* qmap = (const float*)(base + qi.offset);

                int out_dim = mlp_out_dims[m];
                int in_dim = mlp_in_dims[m];
                int total_params = out_dim * in_dim;
                int block_size = 64;
                int n_blocks = total_params / block_size;

                // Step 1: dequantize absmax (uint8 -> float via nested lookup)
                std::vector<float> absmax_f(n_blocks);
                int nested_block_size = 256;
                for (int b = 0; b < n_blocks; b++) {
                    int nested_group = b / nested_block_size;
                    float nested_scale = nested_absmax[nested_group];
                    float dequant_val = nested_qmap[absmax_u8[b]];
                    absmax_f[b] = dequant_val * nested_scale;
                }

                // Step 2: dequantize NF4 weights to fp16
                std::vector<uint16_t> fp16_data(total_params);
                for (int j = 0; j < total_params; j += 2) {
                    int byte_idx = j / 2;
                    uint8_t packed_byte = packed[byte_idx];
                    uint8_t lo = packed_byte & 0x0F;
                    uint8_t hi = (packed_byte >> 4) & 0x0F;

                    int block_lo = j / block_size;
                    int block_hi = (j + 1) / block_size;

                    float w0 = qmap[lo] * absmax_f[block_lo];
                    float w1 = qmap[hi] * absmax_f[block_hi];

                    // Convert float -> fp16
                    fp16_data[j] = float_to_fp16(w0);
                    fp16_data[j + 1] = float_to_fp16(w1);
                }

                // Upload to GPU
                half* gpu_ptr;
                cudaMalloc(reinterpret_cast<void**>(&gpu_ptr), total_params * sizeof(half));
                cudaMemcpy(gpu_ptr, fp16_data.data(), total_params * sizeof(half), cudaMemcpyHostToDevice);
                *mlp_ptrs[m] = gpu_ptr;
            } else {
                // Already fp16/bf16
                *mlp_ptrs[m] = load_half(wname);
            }
        }

        // Norms
        layer.input_layernorm = load_half(p + "input_layernorm.weight");
        layer.post_attn_layernorm = load_half(p + "post_attention_layernorm.weight");

        // QKNorm
        layer.q_norm = load_half(p + "self_attn.q_norm.weight");
        layer.k_norm = load_half(p + "self_attn.k_norm.weight");

        // LoRA (not loaded here)
        layer.lora_q = layer.lora_k = layer.lora_v = layer.lora_o = nullptr;
        layer.lora_gate = layer.lora_up = layer.lora_down = nullptr;

        loaded_layers++;
        if (i == 0 || i == NUM_LAYERS - 1) {
            std::cout << "  Layer " << i << " loaded" << std::endl;
        }
    }
    std::cout << "  All " << loaded_layers << " layers loaded" << std::endl;
}

void InferenceEngine::load_lora(const std::string& lora_prefix, float scale) {
    std::string idx_path = lora_prefix + ".idx";
    std::string bin_path = lora_prefix + ".bin";

    std::cout << "Loading LoRA: " << lora_prefix << std::endl;

    auto index = load_index(idx_path);
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open: " + bin_path);
        return;
    }
    size_t file_size = f.tellg();
    f.seekg(0);
    std::vector<char> data(file_size);
    f.read(data.data(), file_size);
    f.close();

    const char* base = data.data();

    auto load_half = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) return nullptr;
        return (half*)gpu_alloc_copy(base + it->second.offset, it->second.nbytes);
    };

    auto get_shape = [&](const std::string& name, int dim) -> int {
        auto it = index.find(name);
        if (it == index.end() || dim >= (int)it->second.shape.size()) return 0;
        return it->second.shape[dim];
    };

    int loaded = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& layer = weights_.layers[i];
        std::string p = "base_model.model.model.layers." + std::to_string(i) + ".";

        struct { const char* proj; LoRAAdapter** ptr; } targets[] = {
            {"self_attn.q_proj", &layer.lora_q},
            {"self_attn.k_proj", &layer.lora_k},
            {"self_attn.v_proj", &layer.lora_v},
            {"self_attn.o_proj", &layer.lora_o},
            {"mlp.gate_proj", &layer.lora_gate},
            {"mlp.up_proj", &layer.lora_up},
            {"mlp.down_proj", &layer.lora_down},
        };

        for (auto& [proj, ptr] : targets) {
            std::string a_name = p + proj + ".lora_A.weight";
            std::string b_name = p + proj + ".lora_B.weight";
            half* a = load_half(a_name);
            half* b = load_half(b_name);
            if (a && b) {
                auto* adapter = new LoRAAdapter();
                adapter->A = a;
                adapter->B = b;
                adapter->rank = get_shape(a_name, 0);
                adapter->in_features = get_shape(a_name, 1);
                adapter->out_features = get_shape(b_name, 0);
                adapter->scale = scale;
                *ptr = adapter;
                loaded++;
            }
        }
    }
    std::cout << "  Loaded " << loaded << " LoRA adapters" << std::endl;
}
