/**
 * GGUF file format loader.
 *
 * Parses the GGUF header, metadata, and tensor info, then dequantizes
 * quantized tensors to fp16 for use with the CUDA inference engine.
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */
#include "gguf_loader.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

// ── File reading helpers ──

static bool read_bytes(FILE* f, void* dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static bool read_u32(FILE* f, uint32_t& v) { return read_bytes(f, &v, 4); }
static bool read_u64(FILE* f, uint64_t& v) { return read_bytes(f, &v, 8); }
static bool read_i32(FILE* f, int32_t& v)  { return read_bytes(f, &v, 4); }
static bool read_f32(FILE* f, float& v)    { return read_bytes(f, &v, 4); }

static bool read_string(FILE* f, std::string& s) {
    uint64_t len;
    if (!read_u64(f, len)) return false;
    s.resize(len);
    return read_bytes(f, s.data(), len);
}

// Skip a metadata value (for types we don't store)
static bool skip_value(FILE* f, GGUFValueType type);

static bool skip_value(FILE* f, GGUFValueType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            fseek(f, 1, SEEK_CUR); return true;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            fseek(f, 2, SEEK_CUR); return true;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            fseek(f, 4, SEEK_CUR); return true;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            fseek(f, 8, SEEK_CUR); return true;
        case GGUF_TYPE_STRING: {
            uint64_t len;
            if (!read_u64(f, len)) return false;
            fseek(f, len, SEEK_CUR);
            return true;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t count;
            if (!read_u32(f, elem_type) || !read_u64(f, count)) return false;
            for (uint64_t i = 0; i < count; i++) {
                if (!skip_value(f, (GGUFValueType)elem_type)) return false;
            }
            return true;
        }
        default: return false;
    }
}

// ── GGUFFile implementation ──

bool GGUFFile::open(const std::string& path) {
    path_ = path;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "GGUF: cannot open %s\n", path.c_str());
        return false;
    }

    // Header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    read_u32(f, magic);
    read_u32(f, version);
    read_u64(f, n_tensors);
    read_u64(f, n_kv);

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        fclose(f);
        return false;
    }
    if (version < 2 || version > 3) {
        fprintf(stderr, "GGUF: unsupported version %u\n", version);
        fclose(f);
        return false;
    }

    // Read metadata key-value pairs
    metadata_.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; i++) {
        GGUFMetadata m;
        if (!read_string(f, m.key)) { fclose(f); return false; }
        uint32_t vtype;
        if (!read_u32(f, vtype)) { fclose(f); return false; }
        m.type = (GGUFValueType)vtype;

        switch (m.type) {
            case GGUF_TYPE_UINT32: read_u32(f, m.val_uint32); break;
            case GGUF_TYPE_INT32:  read_i32(f, m.val_int32); break;
            case GGUF_TYPE_FLOAT32: read_f32(f, m.val_float32); break;
            case GGUF_TYPE_UINT64: read_u64(f, m.val_uint64); break;
            case GGUF_TYPE_STRING: read_string(f, m.val_string); break;
            case GGUF_TYPE_BOOL: {
                uint8_t b; read_bytes(f, &b, 1);
                m.val_uint32 = b;
                break;
            }
            default:
                // Skip unsupported types (arrays, etc.)
                skip_value(f, m.type);
                break;
        }
        meta_index_[m.key] = metadata_.size();
        metadata_.push_back(std::move(m));
    }

    // Read tensor info entries
    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo t;
        if (!read_string(f, t.name)) { fclose(f); return false; }
        if (!read_u32(f, t.n_dims)) { fclose(f); return false; }

        t.n_elements = 1;
        for (uint32_t d = 0; d < t.n_dims; d++) {
            read_u64(f, t.dims[d]);
            t.n_elements *= t.dims[d];
        }
        for (uint32_t d = t.n_dims; d < 4; d++) t.dims[d] = 1;

        uint32_t ttype;
        read_u32(f, ttype);
        t.type = (GGMLType)ttype;
        read_u64(f, t.offset);

        // Compute data size
        int bs = ggml_block_size(t.type);
        if (bs > 1) {
            uint64_t n_blocks = (t.n_elements + bs - 1) / bs;
            t.data_size = n_blocks * ggml_type_size(t.type);
        } else {
            t.data_size = t.n_elements * ggml_type_size(t.type);
        }

        tensor_index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    // Tensor data starts at the next alignment boundary after header
    // GGUF uses alignment = 32 by default (or from metadata)
    uint64_t alignment = 32;
    auto it = meta_index_.find("general.alignment");
    if (it != meta_index_.end()) {
        alignment = metadata_[it->second].val_uint32;
    }

    uint64_t header_end = ftell(f);
    tensor_data_offset_ = (header_end + alignment - 1) & ~(alignment - 1);

    fclose(f);
    fprintf(stderr, "  GGUF: %zu tensors, %zu metadata entries\n", tensors_.size(), metadata_.size());
    return true;
}

// ── Metadata access ──

int GGUFFile::get_int(const std::string& key, int default_val) const {
    auto it = meta_index_.find(key);
    if (it == meta_index_.end()) return default_val;
    auto& m = metadata_[it->second];
    if (m.type == GGUF_TYPE_UINT32) return (int)m.val_uint32;
    if (m.type == GGUF_TYPE_INT32)  return m.val_int32;
    if (m.type == GGUF_TYPE_UINT64) return (int)m.val_uint64;
    return default_val;
}

float GGUFFile::get_float(const std::string& key, float default_val) const {
    auto it = meta_index_.find(key);
    if (it == meta_index_.end()) return default_val;
    auto& m = metadata_[it->second];
    if (m.type == GGUF_TYPE_FLOAT32) return m.val_float32;
    return default_val;
}

std::string GGUFFile::get_string(const std::string& key, const std::string& default_val) const {
    auto it = meta_index_.find(key);
    if (it == meta_index_.end()) return default_val;
    auto& m = metadata_[it->second];
    if (m.type == GGUF_TYPE_STRING) return m.val_string;
    return default_val;
}

const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

// ── Dequantization routines ──
// These run on CPU, result is uploaded to GPU by load_tensor_fp16.

void GGUFFile::dequant_f32(const void* src, half* dst, uint64_t n) const {
    const float* s = (const float*)src;
    for (uint64_t i = 0; i < n; i++)
        dst[i] = __float2half(s[i]);
}

void GGUFFile::dequant_bf16(const void* src, half* dst, uint64_t n) const {
    const uint16_t* s = (const uint16_t*)src;
    for (uint64_t i = 0; i < n; i++) {
        // bf16 to float: shift left 16 bits
        uint32_t f32_bits = (uint32_t)s[i] << 16;
        float val;
        memcpy(&val, &f32_bits, 4);
        dst[i] = __float2half(val);
    }
}

// Q4_0: 32 elements per block. Block = fp16 scale (2 bytes) + 16 bytes of nibbles
void GGUFFile::dequant_q4_0(const void* src, half* dst, uint64_t n) const {
    const uint8_t* data = (const uint8_t*)src;
    uint64_t n_blocks = n / 32;
    const size_t block_size = 2 + 16;  // fp16 scale + 16 bytes nibbles

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = data + b * block_size;
        // Scale is stored as fp16
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        half scale_h;
        memcpy(&scale_h, &scale_bits, 2);
        float scale = __half2float(scale_h);

        const uint8_t* qs = block + 2;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4) - 8;
            dst[b * 32 + j]      = __float2half(lo * scale);
            dst[b * 32 + j + 16] = __float2half(hi * scale);
        }
    }
}

// Q8_0: 32 elements per block. Block = fp16 scale (2 bytes) + 32 bytes of int8
void GGUFFile::dequant_q8_0(const void* src, half* dst, uint64_t n) const {
    const uint8_t* data = (const uint8_t*)src;
    uint64_t n_blocks = n / 32;
    const size_t block_size = 2 + 32;  // fp16 scale + 32 int8 values

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = data + b * block_size;
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        half scale_h;
        memcpy(&scale_h, &scale_bits, 2);
        float scale = __half2float(scale_h);

        const int8_t* qs = (const int8_t*)(block + 2);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = __float2half(qs[j] * scale);
        }
    }
}

// Q4_K: 256 elements per super-block
// Block layout: fp16 d (2) + fp16 dmin (2) + uint8 scales[12] + uint8 qs[128]
void GGUFFile::dequant_q4_k(const void* src, half* dst, uint64_t n) const {
    const uint8_t* data = (const uint8_t*)src;
    uint64_t n_blocks = n / 256;
    const size_t block_size = 144;  // 2 + 2 + 12 + 128

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = data + b * block_size;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        float d = __half2float(*(half*)&d_bits);
        float dmin = __half2float(*(half*)&dmin_bits);

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        // K-quant scales: 8 sub-blocks of 32 elements each
        // scales[0..5]: 6-bit scales packed, scales[6..11]: 6-bit mins packed
        // Simplified: extract scale and min for each of 8 sub-blocks
        float sc[8], mn[8];
        for (int i = 0; i < 8; i++) {
            if (i < 4) {
                sc[i] = d * (scales[i] & 63);
                mn[i] = dmin * (scales[i + 4] & 63);
            } else {
                sc[i] = d * ((scales[i + 4] & 0xF) | ((scales[i - 4] >> 6) << 4));
                mn[i] = dmin * ((scales[i + 4] >> 4) | ((scales[i] >> 6) << 4));
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 16; j++) {
                uint8_t byte = qs[i * 16 + j];
                int lo = byte & 0x0F;
                int hi = byte >> 4;
                dst[b * 256 + i * 32 + j]      = __float2half(lo * sc[i] - mn[i]);
                dst[b * 256 + i * 32 + j + 16] = __float2half(hi * sc[i] - mn[i]);
            }
        }
    }
}

// Q6_K: 256 elements per super-block
// Block layout: uint8 ql[128] + uint8 qh[64] + int8 scales[16] + fp16 d (2)
void GGUFFile::dequant_q6_k(const void* src, half* dst, uint64_t n) const {
    const uint8_t* data = (const uint8_t*)src;
    uint64_t n_blocks = n / 256;
    const size_t block_size = 210;  // 128 + 64 + 16 + 2

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = data + b * block_size;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* sc = (const int8_t*)(block + 192);

        uint16_t d_bits;
        memcpy(&d_bits, block + 208, 2);
        float d = __half2float(*(half*)&d_bits);

        for (int i = 0; i < 256; i++) {
            int il = i % 128;
            int is = i / 16;

            uint8_t q_lo = ql[il];
            uint8_t q_hi = qh[il / 2];

            int q;
            if (i < 128) {
                q = (q_lo & 0xF) | (((q_hi >> (2 * (il % 2))) & 3) << 4);
            } else {
                q = (q_lo >> 4) | (((q_hi >> (2 * (il % 2) + 4)) & 3) << 4);
            }
            q -= 32;

            dst[b * 256 + i] = __float2half(d * sc[is] * q);
        }
    }
}

// Q5_K: 256 elements per super-block
// Block layout: fp16 d (2) + fp16 dmin (2) + uint8 scales[12] + uint8 qh[32] + uint8 qs[128]
void GGUFFile::dequant_q5_k(const void* src, half* dst, uint64_t n) const {
    const uint8_t* data = (const uint8_t*)src;
    uint64_t n_blocks = n / 256;
    const size_t block_size = 176;  // 2 + 2 + 12 + 32 + 128

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = data + b * block_size;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        float d = __half2float(*(half*)&d_bits);
        float dmin = __half2float(*(half*)&dmin_bits);

        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        float sc[8], mn[8];
        for (int i = 0; i < 8; i++) {
            if (i < 4) {
                sc[i] = d * (scales[i] & 63);
                mn[i] = dmin * (scales[i + 4] & 63);
            } else {
                sc[i] = d * ((scales[i + 4] & 0xF) | ((scales[i - 4] >> 6) << 4));
                mn[i] = dmin * ((scales[i + 4] >> 4) | ((scales[i] >> 6) << 4));
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 16; j++) {
                int idx = i * 32 + j;
                uint8_t byte = qs[i * 16 + j];
                int lo = byte & 0x0F;
                int hi = byte >> 4;
                // 5th bit from qh
                int qh_byte_lo = qh[(i * 32 + j) / 8];
                int qh_byte_hi = qh[(i * 32 + j + 16) / 8];
                int bit_lo = (qh_byte_lo >> ((i * 32 + j) % 8)) & 1;
                int bit_hi = (qh_byte_hi >> ((i * 32 + j + 16) % 8)) & 1;
                lo |= (bit_lo << 4);
                hi |= (bit_hi << 4);
                dst[b * 256 + idx]      = __float2half(lo * sc[i] - mn[i]);
                dst[b * 256 + idx + 16] = __float2half(hi * sc[i] - mn[i]);
            }
        }
    }
}

// ── Load tensor to GPU ──

uint64_t GGUFFile::load_tensor_fp16(const std::string& name, half* gpu_dst) const {
    const GGUFTensorInfo* t = find_tensor(name);
    if (!t) return 0;

    FILE* f = fopen(path_.c_str(), "rb");
    if (!f) return 0;

    // Seek to tensor data
    fseek(f, tensor_data_offset_ + t->offset, SEEK_SET);

    // Read raw data
    std::vector<uint8_t> raw(t->data_size);
    if (fread(raw.data(), 1, t->data_size, f) != t->data_size) {
        fclose(f);
        return 0;
    }
    fclose(f);

    // Dequantize to fp16 on CPU
    std::vector<half> fp16(t->n_elements);

    switch (t->type) {
        case GGML_TYPE_F16:
            // Direct copy
            memcpy(fp16.data(), raw.data(), t->n_elements * sizeof(half));
            break;
        case GGML_TYPE_F32:
            dequant_f32(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_BF16:
            dequant_bf16(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_Q4_0:
            dequant_q4_0(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_Q8_0:
            dequant_q8_0(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_Q4_K:
            dequant_q4_k(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_Q5_K:
            dequant_q5_k(raw.data(), fp16.data(), t->n_elements);
            break;
        case GGML_TYPE_Q6_K:
            dequant_q6_k(raw.data(), fp16.data(), t->n_elements);
            break;
        default:
            fprintf(stderr, "GGUF: unsupported tensor type %d for '%s'\n", (int)t->type, name.c_str());
            return 0;
    }

    // Upload to GPU
    cudaMemcpy(gpu_dst, fp16.data(), t->n_elements * sizeof(half), cudaMemcpyHostToDevice);

    return t->n_elements;
}
