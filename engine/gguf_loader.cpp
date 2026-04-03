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
        // 6-bit scales packed in scales[0..5], 6-bit mins in scales[6..11]
        // Correct extraction: 4 values per 3 bytes
        float sc[8], mn[8];
        // Scales (bytes 0-5)
        sc[0] = scales[0] & 0x3F;
        sc[1] = ((scales[1] & 0x0F) << 2) | (scales[0] >> 6);
        sc[2] = ((scales[2] & 0x03) << 4) | (scales[1] >> 4);
        sc[3] = scales[2] >> 2;
        sc[4] = scales[3] & 0x3F;
        sc[5] = ((scales[4] & 0x0F) << 2) | (scales[3] >> 6);
        sc[6] = ((scales[5] & 0x03) << 4) | (scales[4] >> 4);
        sc[7] = scales[5] >> 2;
        // Mins (bytes 6-11)
        mn[0] = scales[6] & 0x3F;
        mn[1] = ((scales[7] & 0x0F) << 2) | (scales[6] >> 6);
        mn[2] = ((scales[8] & 0x03) << 4) | (scales[7] >> 4);
        mn[3] = scales[8] >> 2;
        mn[4] = scales[9] & 0x3F;
        mn[5] = ((scales[10] & 0x0F) << 2) | (scales[9] >> 6);
        mn[6] = ((scales[11] & 0x03) << 4) | (scales[10] >> 4);
        mn[7] = scales[11] >> 2;
        // Apply d and dmin multipliers
        for (int i = 0; i < 8; i++) {
            sc[i] = d * sc[i];
            mn[i] = dmin * mn[i];
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
    if (!f) {
        fprintf(stderr, "GGUF: cannot reopen %s\n", path_.c_str());
        return 0;
    }

    // Seek to tensor data
    if (fseek(f, tensor_data_offset_ + t->offset, SEEK_SET) != 0) {
        fprintf(stderr, "GGUF: seek failed for '%s'\n", name.c_str());
        fclose(f);
        return 0;
    }

    // Read raw data
    std::vector<uint8_t> raw(t->data_size);
    size_t n_read = fread(raw.data(), 1, t->data_size, f);
    fclose(f);
    if (n_read != t->data_size) {
        fprintf(stderr, "GGUF: truncated read for '%s' (expected %zu, got %zu)\n",
                name.c_str(), (size_t)t->data_size, n_read);
        return 0;
    }

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

    // Upload to GPU with error checking
    cudaError_t err = cudaMemcpy(gpu_dst, fp16.data(), t->n_elements * sizeof(half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GGUF: cudaMemcpy failed for '%s': %s\n", name.c_str(), cudaGetErrorString(err));
        return 0;
    }

    return t->n_elements;
}

// ── Load quantized tensor and convert to Q4L format on GPU ──
// This keeps the base model quantized for QLoRA training.
// Q4L format: packed uint8 data (nibble = value - 8) + per-64-element absmax scale.
uint64_t GGUFFile::load_tensor_q4l(const std::string& name, uint8_t* q4l_data, float* absmax, uint64_t max_elements) const {
    const GGUFTensorInfo* t = find_tensor(name);
    if (!t) return 0;

    FILE* f = fopen(path_.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "GGUF: cannot reopen %s\n", path_.c_str());
        return 0;
    }

    if (fseek(f, tensor_data_offset_ + t->offset, SEEK_SET) != 0) {
        fprintf(stderr, "GGUF: seek failed for '%s'\n", name.c_str());
        fclose(f);
        return 0;
    }

    std::vector<uint8_t> raw(t->data_size);
    size_t n_read = fread(raw.data(), 1, t->data_size, f);
    fclose(f);
    if (n_read != t->data_size) {
        fprintf(stderr, "GGUF: truncated read for '%s'\n", name.c_str());
        return 0;
    }

    uint64_t n = t->n_elements;
    if (n > max_elements) {
        fprintf(stderr, "GGUF: tensor '%s' has %lu elements, max is %lu\n", name.c_str(), (unsigned long)n, (unsigned long)max_elements);
        return 0;
    }

    // Q4L requires block_size=64. GGML types use block_size=32 (Q4_0, Q8_0) or 256 (Q4_K, Q6_K).
    // Strategy: dequant to fp32 on CPU, then re-quantize to Q4L on CPU, then upload.
    // This is a one-time cost during model load.
    std::vector<float> fp32(n);

    switch (t->type) {
        case GGML_TYPE_F16: {
            const uint16_t* src = (const uint16_t*)raw.data();
            for (uint64_t i = 0; i < n; i++) {
                uint16_t h = src[i];
                // Fast fp16 -> fp32
                uint32_t sign = (h >> 15) & 1;
                int32_t exp = ((h >> 10) & 0x1F) - 15;
                int32_t mant = h & 0x3FF;
                float v = (mant / 1024.0f + 1.0f) * (1 << exp) * (sign ? -1.0f : 1.0f);
                if (exp == -15) v = (mant / 1024.0f) * (1.0f / 32768.0f) * (sign ? -1.0f : 1.0f);
                fp32[i] = v;
            }
            break;
        }
        case GGML_TYPE_F32:
            memcpy(fp32.data(), raw.data(), n * sizeof(float));
            break;
        case GGML_TYPE_BF16: {
            const uint16_t* src = (const uint16_t*)raw.data();
            for (uint64_t i = 0; i < n; i++) {
                uint32_t bf = (uint32_t)src[i] << 16;
                fp32[i] = *(float*)&bf;
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            // Block size 32: fp16 scale + 16 bytes packed nibbles
            const size_t block_bytes = 18;
            uint64_t n_blocks = n / 32;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const uint8_t* blk = raw.data() + b * block_bytes;
                uint16_t d_bits;
                memcpy(&d_bits, blk, 2);
                float d = __half2float(*(half*)&d_bits);
                for (int i = 0; i < 16; i++) {
                    uint8_t byte = blk[2 + i];
                    fp32[b * 32 + i]      = ((int8_t)(byte & 0x0F) - 8) * d;
                    fp32[b * 32 + i + 16] = ((int8_t)(byte >> 4) - 8) * d;
                }
            }
            break;
        }
        case GGML_TYPE_Q4_K: {
            const size_t block_bytes = 144;
            uint64_t n_blocks = n / 256;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const uint8_t* block = raw.data() + b * block_bytes;
                uint16_t d_bits, dmin_bits;
                memcpy(&d_bits, block, 2);
                memcpy(&dmin_bits, block + 2, 2);
                float d = __half2float(*(half*)&d_bits);
                float dmin = __half2float(*(half*)&dmin_bits);
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;
                float sc[8], mn[8];
                sc[0] = d * (scales[0] & 0x3F);
                sc[1] = d * (((scales[1] & 0x0F) << 2) | (scales[0] >> 6));
                sc[2] = d * (((scales[2] & 0x03) << 4) | (scales[1] >> 4));
                sc[3] = d * (scales[2] >> 2);
                sc[4] = d * (scales[3] & 0x3F);
                sc[5] = d * (((scales[4] & 0x0F) << 2) | (scales[3] >> 6));
                sc[6] = d * (((scales[5] & 0x03) << 4) | (scales[4] >> 4));
                sc[7] = d * (scales[5] >> 2);
                mn[0] = dmin * (scales[6] & 0x3F);
                mn[1] = dmin * (((scales[7] & 0x0F) << 2) | (scales[6] >> 6));
                mn[2] = dmin * (((scales[8] & 0x03) << 4) | (scales[7] >> 4));
                mn[3] = dmin * (scales[8] >> 2);
                mn[4] = dmin * (scales[9] & 0x3F);
                mn[5] = dmin * (((scales[10] & 0x0F) << 2) | (scales[9] >> 6));
                mn[6] = dmin * (((scales[11] & 0x03) << 4) | (scales[10] >> 4));
                mn[7] = dmin * (scales[11] >> 2);
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 16; j++) {
                        uint8_t byte = qs[i * 16 + j];
                        fp32[b * 256 + i * 32 + j]      = (byte & 0x0F) * sc[i] - mn[i];
                        fp32[b * 256 + i * 32 + j + 16] = (byte >> 4) * sc[i] - mn[i];
                    }
                }
            }
            break;
        }
        case GGML_TYPE_Q8_0: {
            const size_t block_bytes = 34;
            uint64_t n_blocks = n / 32;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const uint8_t* blk = raw.data() + b * block_bytes;
                uint16_t d_bits;
                memcpy(&d_bits, blk, 2);
                float d = __half2float(*(half*)&d_bits);
                for (int i = 0; i < 32; i++) {
                    fp32[b * 32 + i] = ((int8_t)blk[2 + i]) * d;
                }
            }
            break;
        }
        case GGML_TYPE_Q6_K: {
            const size_t block_bytes = 210;
            uint64_t n_blocks = n / 256;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const uint8_t* blk = raw.data() + b * block_bytes;
                const uint8_t* ql = blk;
                const uint8_t* qh = blk + 128;
                const int8_t* scales = (const int8_t*)(blk + 192);
                uint16_t d_bits;
                memcpy(&d_bits, blk + 208, 2);
                float d = __half2float(*(half*)&d_bits);
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        int idx = i * 16 + j;
                        int ql_val = ql[idx];
                        int qh_bits = (qh[i * 2 + (j / 8)] >> ((j % 8) * 2)) & 0x03;
                        int q = (ql_val & 0x0F) | (qh_bits << 4);
                        fp32[b * 256 + i * 16 + j] = (q - 32) * scales[i] * d;
                    }
                }
            }
            break;
        }
        default:
            fprintf(stderr, "GGUF: unsupported type %d for Q4L conversion of '%s'\n", (int)t->type, name.c_str());
            return 0;
    }

    // Re-quantize fp32 -> Q4L format with dp4a-friendly packing (block_size=64)
    //
    // dp4a packing: for each group of 8 elements [e0..e7]:
    //   byte[0] = e0 | (e4 << 4)
    //   byte[1] = e1 | (e5 << 4)
    //   byte[2] = e2 | (e6 << 4)
    //   byte[3] = e3 | (e7 << 4)
    //
    // dp4a extraction:
    //   (packed >> 0) & 0x0F0F0F0F = [e0, e1, e2, e3]
    //   (packed >> 4) & 0x0F0F0F0F = [e4, e5, e6, e7]
    int block_size = 64;
    int n_blocks = (n + block_size - 1) / block_size;

    std::vector<uint8_t> q4l_packed(n_blocks * block_size / 2);  // 4 bits per element
    std::vector<float> q4l_absmax(n_blocks);

    for (int b = 0; b < n_blocks; b++) {
        int base = b * block_size;
        // Find absmax for this block
        float amax = 0.0f;
        for (int i = 0; i < block_size && base + i < (int)n; i++) {
            float v = fabsf(fp32[base + i]);
            if (v > amax) amax = v;
        }
        if (amax < 1e-10f) amax = 1e-10f;
        q4l_absmax[b] = amax / 7.5f;  // scale = absmax / 7.5 (matches convert_weights.py)

        float inv_scale = 7.5f / amax;
        // Quantize to nibbles
        uint8_t nibbles[64] = {};
        for (int i = 0; i < block_size && base + i < (int)n; i++) {
            int q = (int)roundf(fp32[base + i] * inv_scale + 8.0f);
            nibbles[i] = (uint8_t)std::max(0, std::min(15, q));
        }
        // dp4a-friendly interleave: groups of 8
        uint8_t* dst = q4l_packed.data() + b * (block_size / 2);
        for (int g = 0; g < block_size / 8; g++) {
            uint8_t* grp = nibbles + g * 8;
            dst[g * 4 + 0] = grp[0] | (grp[4] << 4);
            dst[g * 4 + 1] = grp[1] | (grp[5] << 4);
            dst[g * 4 + 2] = grp[2] | (grp[6] << 4);
            dst[g * 4 + 3] = grp[3] | (grp[7] << 4);
        }
    }

    // Upload to GPU
    size_t data_bytes = (size_t)n_blocks * block_size / 2;
    size_t scale_bytes = (size_t)n_blocks * sizeof(float);
    cudaError_t err;
    err = cudaMemcpy(q4l_data, q4l_packed.data(), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GGUF: cudaMemcpy failed for q4l_data '%s': %s\n", name.c_str(), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMemcpy(absmax, q4l_absmax.data(), scale_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GGUF: cudaMemcpy failed for absmax '%s': %s\n", name.c_str(), cudaGetErrorString(err));
        return 0;
    }

    return n;
}
