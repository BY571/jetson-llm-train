/**
 * CUDA kernels for Jetson LLM inference engine.
 *
 * Optimized for Jetson Orin sm_87 (Ampere, 32 tensor cores).
 * Single-batch decode (batch_size=1), which means GEMV not GEMM.
 *
 * Key optimization: fuse operations to reduce kernel launch overhead.
 * The profiler showed 1400 kernel launches per token with bitsandbytes;
 * we target ~10-15 kernel launches per token.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "model.h"

// ============================================================================
// NF4 Dequantization Lookup Table (standard bitsandbytes NF4 values)
// ============================================================================

__constant__ float NF4_LOOKUP[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// ============================================================================
// Kernel 1: Fused NF4 Dequant + GEMV (Matrix-Vector multiply)
// ============================================================================
// Replaces 196 separate bitsandbytes kernel calls per token.
// For decode: input is (1, in_dim), weight is (out_dim, in_dim) in NF4.
// Output: (1, out_dim) in fp16.
//
// Each thread block computes one output element.
// Threads in the block cooperatively dequantize and dot-product.

__global__ void nf4_gemv_kernel(
    const uint8_t* __restrict__ weight_data,    // NF4 packed (2 values per byte)
    const half* __restrict__ absmax,             // per-block scales
    const half* __restrict__ input,              // (in_dim,) fp16
    half* __restrict__ output,                   // (out_dim,) fp16
    int in_dim,
    int out_dim,
    int block_size                               // quantization block size (64)
) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    // Each output element = dot(weight_row[out_idx], input)
    // weight_row is (in_dim,) stored as NF4

    const uint8_t* row = weight_data + (size_t)out_idx * (in_dim / 2);
    int num_blocks = (in_dim + block_size - 1) / block_size;

    float sum = 0.0f;

    // Each thread handles a chunk of the dot product
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float scale = __half2float(absmax[out_idx * num_blocks + b]);
        int start = b * block_size;
        int end = min(start + block_size, in_dim);

        float local_sum = 0.0f;
        for (int j = start; j < end; j += 2) {
            int byte_idx = j / 2;
            uint8_t packed = row[byte_idx];
            uint8_t lo = packed & 0x0F;
            uint8_t hi = (packed >> 4) & 0x0F;

            float w0 = NF4_LOOKUP[lo] * scale;
            float w1 = NF4_LOOKUP[hi] * scale;
            float x0 = __half2float(input[j]);
            float x1 = (j + 1 < end) ? __half2float(input[j + 1]) : 0.0f;

            local_sum += w0 * x0 + w1 * x1;
        }
        sum += local_sum;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Block-level reduction (first warp collects)
    __shared__ float shared_sum[32]; // max 32 warps
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            output[out_idx] = __float2half(sum);
        }
    }
}

// With LoRA: output = (base_weight @ input) + scale * (B @ (A @ input))
__global__ void nf4_gemv_lora_kernel(
    const uint8_t* __restrict__ weight_data,
    const half* __restrict__ absmax,
    const half* __restrict__ input,
    const half* __restrict__ lora_A,    // (rank, in_dim)
    const half* __restrict__ lora_B,    // (out_dim, rank)
    half* __restrict__ output,
    half* __restrict__ lora_scratch,    // (rank,) intermediate
    int in_dim,
    int out_dim,
    int block_size,
    int rank,
    float lora_scale
) {
    // First compute base: same as nf4_gemv_kernel
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    const uint8_t* row = weight_data + (size_t)out_idx * (in_dim / 2);
    int num_blocks = (in_dim + block_size - 1) / block_size;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float scale = __half2float(absmax[out_idx * num_blocks + b]);
        int start = b * block_size;
        int end = min(start + block_size, in_dim);

        float local_sum = 0.0f;
        for (int j = start; j < end; j += 2) {
            int byte_idx = j / 2;
            uint8_t packed = row[byte_idx];
            float w0 = NF4_LOOKUP[packed & 0x0F] * scale;
            float w1 = NF4_LOOKUP[(packed >> 4) & 0x0F] * scale;
            local_sum += w0 * __half2float(input[j]);
            if (j + 1 < end) local_sum += w1 * __half2float(input[j + 1]);
        }
        sum += local_sum;
    }

    // Reduce sum (same as above)
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared_sum[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lane_id == 0) {
            // Add LoRA contribution: scale * B[out_idx, :] @ lora_scratch
            // lora_scratch was pre-computed as A @ input
            float lora_sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                lora_sum += __half2float(lora_B[out_idx * rank + r]) *
                            __half2float(lora_scratch[r]);
            }
            output[out_idx] = __float2half(sum + lora_scale * lora_sum);
        }
    }
}

// ============================================================================
// Kernel 2: RMSNorm
// ============================================================================
// output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)

__global__ void rms_norm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int dim,
    float eps
) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[i]);
        sum_sq += val * val;
    }

    // Block reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane_id == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / dim + eps);

    // Normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[i]) * rms;
        output[i] = __float2half(val * __half2float(weight[i]));
    }
}

// ============================================================================
// Kernel 3: RoPE (Rotary Position Embedding)
// ============================================================================
// Apply rotation to Q and K vectors in-place.

__global__ void rope_kernel(
    half* __restrict__ q,       // (Q_DIM,) = (NUM_HEADS * HEAD_DIM,)
    half* __restrict__ k,       // (KV_DIM,) = (NUM_KV_HEADS * HEAD_DIM,)
    const half* __restrict__ cos_table,  // (HEAD_DIM/2,) for this position
    const half* __restrict__ sin_table,  // (HEAD_DIM/2,) for this position
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head = head_dim / 2;
    int total_q = num_q_heads * half_head;
    int total_kv = num_kv_heads * half_head;

    // Process Q
    if (idx < total_q) {
        int head = idx / half_head;
        int d = idx % half_head;
        int base = head * head_dim;

        float q0 = __half2float(q[base + d]);
        float q1 = __half2float(q[base + d + half_head]);
        float c = __half2float(cos_table[d]);
        float s = __half2float(sin_table[d]);

        q[base + d] = __float2half(q0 * c - q1 * s);
        q[base + d + half_head] = __float2half(q1 * c + q0 * s);
    }

    // Process K
    if (idx < total_kv) {
        int head = idx / half_head;
        int d = idx % half_head;
        int base = head * head_dim;

        float k0 = __half2float(k[base + d]);
        float k1 = __half2float(k[base + d + half_head]);
        float c = __half2float(cos_table[d]);
        float s = __half2float(sin_table[d]);

        k[base + d] = __float2half(k0 * c - k1 * s);
        k[base + d + half_head] = __float2half(k1 * c + k0 * s);
    }
}

// ============================================================================
// Kernel 4: GQA Attention (decode, single token)
// ============================================================================
// For position `pos`, compute attention over KV cache [0..pos].
// GQA: each Q head group shares one KV head.

__global__ void gqa_attention_decode_kernel(
    const half* __restrict__ q,         // (Q_DIM,) current Q
    const half* __restrict__ k_cache,   // (max_seq, KV_DIM) full K cache
    const half* __restrict__ v_cache,   // (max_seq, KV_DIM) full V cache
    half* __restrict__ output,          // (Q_DIM,) attention output
    float* __restrict__ attn_scratch,   // (NUM_HEADS, max_seq) scratch for scores
    int pos,                            // current position (attend to 0..pos inclusive)
    int max_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int head = blockIdx.x; // one block per Q head
    if (head >= num_q_heads) return;

    int kv_head = head / (num_q_heads / num_kv_heads); // GQA mapping
    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute attention scores: q @ k^T for this head
    float* scores = attn_scratch + head * max_seq_len;
    const half* q_head = q + head * head_dim;

    // Score computation: each thread handles some positions
    float max_score = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const half* k_p = k_cache + p * qwen3::KV_DIM + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_head[d]) * __half2float(k_p[d]);
        }
        dot *= scale;
        scores[p] = dot;
        if (dot > max_score) max_score = dot;
    }

    // Block-level max reduction for softmax stability
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));

    __shared__ float shared_max[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_max[warp_id] = max_score;
    __syncthreads();
    if (warp_id == 0) {
        max_score = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane_id] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
        if (lane_id == 0) shared_max[0] = max_score;
    }
    __syncthreads();
    max_score = shared_max[0];

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        float val = expf(scores[p] - max_score);
        scores[p] = val;
        sum_exp += val;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane_id == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (lane_id == 0) shared_sum[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / (shared_sum[0] + 1e-8f);

    // Normalize scores
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        scores[p] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V: output[d] = sum_p(scores[p] * v_cache[p, kv_head, d])
    half* out_head = output + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int p = 0; p <= pos; p++) {
            val += scores[p] * __half2float(v_cache[p * qwen3::KV_DIM + kv_head * head_dim + d]);
        }
        out_head[d] = __float2half(val);
    }
}

// ============================================================================
// Kernel 5: Fused SiLU-Gate-Mul
// ============================================================================
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)

__global__ void silu_gate_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g)); // silu = x * sigmoid(x)
        output[idx] = __float2half(silu_g * u);
    }
}

// ============================================================================
// Kernel 6: Embedding Lookup
// ============================================================================

__global__ void embedding_lookup_kernel(
    const half* __restrict__ embed_table,   // (vocab_size, hidden)
    int token_id,
    half* __restrict__ output,              // (hidden,)
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = embed_table[(size_t)token_id * hidden_dim + idx];
    }
}

// ============================================================================
// Kernel 7: Residual Add
// ============================================================================

__global__ void residual_add_kernel(
    half* __restrict__ output,      // output += residual
    const half* __restrict__ residual,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float o = __half2float(output[idx]);
        float r = __half2float(residual[idx]);
        output[idx] = __float2half(o + r);
    }
}

// ============================================================================
// Kernel 8a: FP16 GEMV (weight @ input -> output, all fp16)
// ============================================================================
// Used for attention projections and dequantized MLP weights.
// One block per output element, threads cooperate on the dot product.

__global__ void fp16_gemv_kernel(
    const half* __restrict__ weight,    // (out_dim, in_dim) row-major
    const half* __restrict__ input,     // (in_dim,)
    half* __restrict__ output,          // (out_dim,)
    int in_dim,
    int out_dim
) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    float sum = 0.0f;
    const half* row = weight + (size_t)out_idx * in_dim;

    for (int j = threadIdx.x; j < in_dim; j += blockDim.x) {
        sum += __half2float(row[j]) * __half2float(input[j]);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) {
            output[out_idx] = __float2half(sum);
        }
    }
}

// ============================================================================
// Kernel 8b: GEMV for lm_head (fp16 weight @ fp16 input -> fp32 logits)
// ============================================================================
// lm_head is tied to embedding, so weights are fp16 (not NF4)

__global__ void fp16_gemv_logits_kernel(
    const half* __restrict__ weight,    // (vocab_size, hidden)
    const half* __restrict__ input,     // (hidden,)
    float* __restrict__ logits,         // (vocab_size,) float32
    int hidden_dim,
    int vocab_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= vocab_size) return;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        sum += __half2float(weight[(size_t)out_idx * hidden_dim + j]) *
               __half2float(input[j]);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) {
            logits[out_idx] = sum;
        }
    }
}

// ============================================================================
// Kernel 9: Top-p sampling
// ============================================================================
// Temperature scaling + softmax + cumulative top-p + sample

__global__ void temperature_scale_kernel(
    float* __restrict__ logits,
    int vocab_size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] /= temperature;
    }
}

// ============================================================================
// Host-side launcher functions
// ============================================================================

extern "C" {

void launch_nf4_gemv(
    const NF4Weight& weight,
    const half* input,
    half* output,
    cudaStream_t stream
) {
    int threads = 128;
    nf4_gemv_kernel<<<weight.cols, threads, 0, stream>>>(
        weight.data, weight.absmax, input, output,
        weight.rows, weight.cols, weight.block_size
    );
}

void launch_rms_norm(
    const half* input,
    const half* weight,
    half* output,
    int dim,
    float eps,
    cudaStream_t stream
) {
    rms_norm_kernel<<<1, 256, 0, stream>>>(input, weight, output, dim, eps);
}

void launch_rope(
    half* q, half* k,
    const half* cos_table, const half* sin_table,
    int pos, int max_seq_len,
    cudaStream_t stream
) {
    int n = qwen3::NUM_HEADS * qwen3::HEAD_DIM / 2;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k,
        cos_table + pos * (qwen3::HEAD_DIM / 2),
        sin_table + pos * (qwen3::HEAD_DIM / 2),
        qwen3::NUM_HEADS, qwen3::NUM_KV_HEADS, qwen3::HEAD_DIM
    );
}

void launch_gqa_attention(
    const half* q,
    const half* k_cache, const half* v_cache,
    half* output,
    float* attn_scratch,
    int pos, int max_seq_len,
    cudaStream_t stream
) {
    // One block per Q head, 128 threads per block
    gqa_attention_decode_kernel<<<qwen3::NUM_HEADS, 128, 0, stream>>>(
        q, k_cache, v_cache, output, attn_scratch,
        pos, max_seq_len,
        qwen3::NUM_HEADS, qwen3::NUM_KV_HEADS, qwen3::HEAD_DIM
    );
}

void launch_silu_gate_mul(
    const half* gate, const half* up, half* output,
    int dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    silu_gate_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, dim);
}

void launch_embedding(
    const half* embed_table, int token_id,
    half* output, int hidden_dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        embed_table, token_id, output, hidden_dim
    );
}

void launch_residual_add(
    half* output, const half* residual,
    int dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(output, residual, dim);
}

void launch_fp16_gemv(
    const half* weight,
    const half* input,
    half* output,
    int out_dim,
    int in_dim,
    cudaStream_t stream
) {
    fp16_gemv_kernel<<<out_dim, 128, 0, stream>>>(
        weight, input, output, in_dim, out_dim
    );
}

void launch_lm_head(
    const half* weight, const half* input,
    float* logits, int hidden_dim, int vocab_size,
    cudaStream_t stream
) {
    fp16_gemv_logits_kernel<<<vocab_size, 128, 0, stream>>>(
        weight, input, logits, hidden_dim, vocab_size
    );
}

} // extern "C"
