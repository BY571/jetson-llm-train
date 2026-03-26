/**
 * NF4 GEMV kernels for Jetson Orin (sm_87) — v7.
 *
 * Two structural optimizations over v6:
 *
 * 1. Warp-shuffle NF4 lookup: each thread holds one of 16 NF4 table values
 *    in a register. Dequantization uses __shfl_sync to read the needed value
 *    from the appropriate lane (~4 cycle latency vs ~20 for shared memory).
 *    Eliminates shared memory NF4 table entirely.
 *
 * 2. Vectorized uint4 input load: loads 8 consecutive half values (16 bytes)
 *    in a single memory transaction, replacing 8 separate 2-byte loads.
 *    Requires 16-byte alignment (guaranteed: base_j * 2 is always multiple of 16).
 *
 * Per-iteration instruction reduction:
 *   v6: 10 global loads + 8 smem lookups (160 cycles) = ~46 ops
 *   v7:  3 global loads + 8 shuffles   ( 32 cycles) = ~42 ops, much lower latency
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// NF4 dequantization table (indexed 0-15)
static __device__ const float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

__device__ __forceinline__ void nf4_gemv_block(
    float nf4_reg,  // this thread's NF4 table value (indexed by lane_id & 0xF)
    float* __restrict__ s_sums,  // [ROWS_PER_BLOCK * N_WARPS] for cross-warp reduction
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int first_row, int out_dim, int in_dim
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;
    const int vec_per_row = bytes_per_row >> 2;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += THREADS_PER_BLOCK) {
            // 1. Load 4 weight bytes (8 NF4 values)
            uint32_t packed4 = __ldg(&row_u32[vi]);
            int base_j = vi << 3;

            // 2. Load absmax (all 8 values share one block, exact)
            float scale = __ldg(&absmax[absmax_row_start + (base_j >> 6)]);

            // 3. Load 8 input values in one 16-byte transaction
            uint4 in_vec = __ldg(reinterpret_cast<const uint4*>(&input[base_j]));
            half2 x01 = *reinterpret_cast<const half2*>(&in_vec.x);
            half2 x23 = *reinterpret_cast<const half2*>(&in_vec.y);
            half2 x45 = *reinterpret_cast<const half2*>(&in_vec.z);
            half2 x67 = *reinterpret_cast<const half2*>(&in_vec.w);

            // 4. Dequant + FMA for each byte (warp shuffle for NF4 lookup)
            // Byte 0: elements 0,1
            uint8_t b0 = packed4 & 0xFF;
            float w0 = __shfl_sync(0xFFFFFFFF, nf4_reg, b0 >> 4) * scale;
            float w1 = __shfl_sync(0xFFFFFFFF, nf4_reg, b0 & 0xF) * scale;
            sum = __fmaf_rn(w0, __half2float(x01.x), sum);
            sum = __fmaf_rn(w1, __half2float(x01.y), sum);

            // Byte 1: elements 2,3
            uint8_t b1 = (packed4 >> 8) & 0xFF;
            float w2 = __shfl_sync(0xFFFFFFFF, nf4_reg, b1 >> 4) * scale;
            float w3 = __shfl_sync(0xFFFFFFFF, nf4_reg, b1 & 0xF) * scale;
            sum = __fmaf_rn(w2, __half2float(x23.x), sum);
            sum = __fmaf_rn(w3, __half2float(x23.y), sum);

            // Byte 2: elements 4,5
            uint8_t b2 = (packed4 >> 16) & 0xFF;
            float w4 = __shfl_sync(0xFFFFFFFF, nf4_reg, b2 >> 4) * scale;
            float w5 = __shfl_sync(0xFFFFFFFF, nf4_reg, b2 & 0xF) * scale;
            sum = __fmaf_rn(w4, __half2float(x45.x), sum);
            sum = __fmaf_rn(w5, __half2float(x45.y), sum);

            // Byte 3: elements 6,7
            uint8_t b3 = (packed4 >> 24) & 0xFF;
            float w6 = __shfl_sync(0xFFFFFFFF, nf4_reg, b3 >> 4) * scale;
            float w7 = __shfl_sync(0xFFFFFFFF, nf4_reg, b3 & 0xF) * scale;
            sum = __fmaf_rn(w6, __half2float(x67.x), sum);
            sum = __fmaf_rn(w7, __half2float(x67.y), sum);
        }

        // Remainder (bytes_per_row not divisible by 4 — not needed for Qwen3 dims)
        int rem_start = vec_per_row << 2;
        for (int bi = rem_start + threadIdx.x; bi < bytes_per_row;
             bi += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + bi]);
            int j0 = bi * 2;
            float sc = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            float wa = __shfl_sync(0xFFFFFFFF, nf4_reg, packed >> 4) * sc;
            float wb = __shfl_sync(0xFFFFFFFF, nf4_reg, packed & 0xF) * sc;
            sum = __fmaf_rn(wa, __half2float(__ldg(&input[j0])), sum);
            sum = __fmaf_rn(wb, __half2float(__ldg(&input[j0 + 1])), sum);
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lane_id == 0) s_sums[r * N_WARPS + warp_id] = sum;
        __syncthreads();

        if (warp_id == 0) {
            float total = (lane_id < N_WARPS) ? s_sums[r * N_WARPS + lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            if (lane_id == 0) output[row] = __float2half(total);
        }
        __syncthreads();
    }
}

// Shared: only cross-warp reduction buffer (128 bytes)
// NF4 table in registers via warp shuffle — no shared memory needed

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ a,
    const half* __restrict__ x, half* __restrict__ y,
    int in_dim, int out_dim, int block_size
) {
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    float nf4_reg = NF4_TABLE[threadIdx.x % 32 & 0xF];
    nf4_gemv_block(nf4_reg, s_sums, w, a, x, y,
                   blockIdx.x * ROWS_PER_BLOCK, out_dim, in_dim);
}

__global__ void nf4_fused_2_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const half* __restrict__ x, int in_dim
) {
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    float nf4_reg = NF4_TABLE[threadIdx.x % 32 & 0xF];
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab)
        nf4_gemv_block(nf4_reg, s_sums, aw, aa, x, ay,
                       blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    else
        nf4_gemv_block(nf4_reg, s_sums, bw, ba, x, by,
                       (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
}

__global__ void nf4_fused_3_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const uint8_t* __restrict__ cw, const float* __restrict__ ca,
    half* __restrict__ cy, int cd,
    const half* __restrict__ x, int in_dim
) {
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    float nf4_reg = NF4_TABLE[threadIdx.x % 32 & 0xF];
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int bb = (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab)
        nf4_gemv_block(nf4_reg, s_sums, aw, aa, x, ay,
                       blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    else if (blockIdx.x < ab + bb)
        nf4_gemv_block(nf4_reg, s_sums, bw, ba, x, by,
                       (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
    else
        nf4_gemv_block(nf4_reg, s_sums, cw, ca, x, cy,
                       (blockIdx.x - ab - bb) * ROWS_PER_BLOCK, cd, in_dim);
}

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* p, const float* a, const half* x, half* y,
    int od, int id, int bs, cudaStream_t s
) {
    int nb = (od + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(p, a, x, y, id, od, bs);
}

void launch_nf4_fused_2(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(
        aw, aa, ay, ad, bw, ba, by, bd, x, id);
}

void launch_nf4_fused_3(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const uint8_t* cw, const float* ca, half* cy, int cd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (cd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_3_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(
        aw, aa, ay, ad, bw, ba, by, bd, cw, ca, cy, cd, x, id);
}

} // extern "C"
