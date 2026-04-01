/**
 * GGUF file format loader for the inference engine.
 *
 * Reads quantized weights from GGUF files (llama.cpp format),
 * dequantizes to fp16 on GPU, and populates engine weight structures.
 *
 * Supported quantization types:
 *   - F16:    no dequant needed
 *   - F32:    cast to fp16
 *   - Q4_0:   4-bit with one scale per 32-element block
 *   - Q4_K_M: 4-bit K-quants with min/scale per super-block
 *   - Q8_0:   8-bit with one scale per 32-element block
 *   - Q6_K:   6-bit K-quants
 *
 * Usage:
 *   GGUFFile gguf("model.gguf");
 *   half* tensor = gguf.load_tensor_fp16("blk.0.attn_q.weight", gpu_ptr);
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_fp16.h>

// GGUF constants
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" as uint32 little-endian: G(47) G(47) U(55) F(46)
constexpr uint32_t GGUF_VERSION = 3;

// GGML tensor types
enum GGMLType : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_BF16    = 30,
};

// GGUF metadata value types
enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Block sizes for quantization types
inline int ggml_block_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q3_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        case GGML_TYPE_Q8_K: return 256;
        default: return 1;
    }
}

// Bytes per block for quantization types
inline size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;   // 2 bytes scale + 16 bytes data (32 nibbles)
        case GGML_TYPE_Q4_1: return 20;   // 2+2 bytes scale/min + 16 bytes data
        case GGML_TYPE_Q5_0: return 22;   // 2 bytes scale + 4 bytes high bits + 16 bytes low
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;   // 2 bytes scale + 32 bytes data
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 256/16*2 + 256/4 + 2 + 2;  // ~84
        case GGML_TYPE_Q3_K: return 256/8*3 + 256/16 + 12 + 2;  // ~110
        case GGML_TYPE_Q4_K: return 2 + 2 + 12 + 256/2;  // 144
        case GGML_TYPE_Q5_K: return 2 + 2 + 12 + 256/8 + 256/2;  // 176
        case GGML_TYPE_Q6_K: return 256/2 + 256/4 + 256/16 + 2;  // 210
        case GGML_TYPE_Q8_K: return 4 + 256 + 16*2;  // 292
        default: return 0;
    }
}

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    GGMLType type;
    uint64_t offset;      // offset from start of tensor data section
    size_t data_size;     // total bytes
    uint64_t n_elements;  // total element count
};

struct GGUFMetadata {
    std::string key;
    GGUFValueType type;
    // Store common types
    uint32_t val_uint32;
    int32_t val_int32;
    float val_float32;
    uint64_t val_uint64;
    std::string val_string;
};

class GGUFFile {
public:
    bool open(const std::string& path);

    // Metadata access
    int get_int(const std::string& key, int default_val = 0) const;
    float get_float(const std::string& key, float default_val = 0.0f) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;

    // Tensor access
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    // Load a tensor, dequantize to fp16 on GPU
    // Caller provides GPU destination pointer (must be pre-allocated: n_elements * sizeof(half))
    // Returns number of elements, or 0 on failure
    uint64_t load_tensor_fp16(const std::string& name, half* gpu_dst) const;

    // Get all tensor names
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    uint64_t tensor_data_offset() const { return tensor_data_offset_; }

private:
    std::string path_;
    std::vector<GGUFTensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;  // name -> index in tensors_
    std::vector<GGUFMetadata> metadata_;
    std::unordered_map<std::string, size_t> meta_index_;
    uint64_t tensor_data_offset_ = 0;

    // Dequantize helpers (CPU, then upload to GPU)
    void dequant_q4_0(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_q8_0(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_q4_k(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_q5_k(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_q6_k(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_f32(const void* src, half* dst, uint64_t n_elements) const;
    void dequant_bf16(const void* src, half* dst, uint64_t n_elements) const;
};
