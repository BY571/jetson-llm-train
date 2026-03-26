/**
 * 4-bit GEMV kernels for Jetson Orin (sm_87) — v8 (linear dequant).
 *
 * Supports two weight formats:
 *   NF4: non-linear, requires shared memory lookup table (18-28 GB/s)
 *   Q4L: linear, dequant = (nibble - 8) * scale, all ALU (targeting 40+ GB/s)
 *
 * Q4L eliminates the shared memory bottleneck entirely. The dequant is just
 * a float conversion + subtract + multiply, all pipelined in the ALU.
 *
 * Architecture: same as v6 (4-byte vectorized loads, 4 rows/block, 256 threads).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// ============================================================================
// Q4 Linear GEMV: dequant = (nibble - 8) * scale (NO lookup table)
// ============================================================================

__device__ __forceinline__ void q4l_gemv_block(
    float* __restrict__ s_sums,
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ scales,  // one fp32 scale per 64-element block
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
        const int scale_row_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += THREADS_PER_BLOCK) {
            uint32_t packed4 = __ldg(&row_u32[vi]);
            int base_j = vi << 3;
            float scale = __ldg(&scales[scale_row_start + (base_j >> 6)]);

            // Byte 0: linear dequant (nibble - 8) * scale — no lookup!
            uint8_t b0 = packed4 & 0xFF;
            float w0 = ((float)(b0 >> 4) - 8.0f) * scale;
            float w1 = ((float)(b0 & 0xF) - 8.0f) * scale;
            sum = __fmaf_rn(w0, __half2float(__ldg(&input[base_j])), sum);
            sum = __fmaf_rn(w1, __half2float(__ldg(&input[base_j + 1])), sum);

            // Byte 1
            uint8_t b1 = (packed4 >> 8) & 0xFF;
            float w2 = ((float)(b1 >> 4) - 8.0f) * scale;
            float w3 = ((float)(b1 & 0xF) - 8.0f) * scale;
            sum = __fmaf_rn(w2, __half2float(__ldg(&input[base_j + 2])), sum);
            sum = __fmaf_rn(w3, __half2float(__ldg(&input[base_j + 3])), sum);

            // Byte 2
            uint8_t b2 = (packed4 >> 16) & 0xFF;
            float w4 = ((float)(b2 >> 4) - 8.0f) * scale;
            float w5 = ((float)(b2 & 0xF) - 8.0f) * scale;
            sum = __fmaf_rn(w4, __half2float(__ldg(&input[base_j + 4])), sum);
            sum = __fmaf_rn(w5, __half2float(__ldg(&input[base_j + 5])), sum);

            // Byte 3
            uint8_t b3 = (packed4 >> 24) & 0xFF;
            float w6 = ((float)(b3 >> 4) - 8.0f) * scale;
            float w7 = ((float)(b3 & 0xF) - 8.0f) * scale;
            sum = __fmaf_rn(w6, __half2float(__ldg(&input[base_j + 6])), sum);
            sum = __fmaf_rn(w7, __half2float(__ldg(&input[base_j + 7])), sum);
        }

        // Remainder
        int rem_start = vec_per_row << 2;
        for (int bi = rem_start + threadIdx.x; bi < bytes_per_row; bi += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + bi]);
            int j0 = bi * 2;
            float sc = __ldg(&scales[scale_row_start + (j0 >> 6)]);
            sum = __fmaf_rn(((float)(packed >> 4) - 8.0f) * sc, __half2float(__ldg(&input[j0])), sum);
            sum = __fmaf_rn(((float)(packed & 0xF) - 8.0f) * sc, __half2float(__ldg(&input[j0 + 1])), sum);
        }

        // Warp + cross-warp reduction
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

// ============================================================================
// NF4 GEMV: non-linear, shared memory lookup table (kept as fallback)
// ============================================================================

__device__ __forceinline__ void nf4_gemv_block(
    const float* __restrict__ s_qmap,
    float* __restrict__ s_sums,
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
            uint32_t packed4 = __ldg(&row_u32[vi]);
            int base_j = vi << 3;
            float scale = __ldg(&absmax[absmax_row_start + (base_j >> 6)]);

            uint8_t b0 = packed4 & 0xFF;
            sum = __fmaf_rn(s_qmap[b0 >> 4] * scale, __half2float(__ldg(&input[base_j])), sum);
            sum = __fmaf_rn(s_qmap[b0 & 0xF] * scale, __half2float(__ldg(&input[base_j + 1])), sum);

            uint8_t b1 = (packed4 >> 8) & 0xFF;
            sum = __fmaf_rn(s_qmap[b1 >> 4] * scale, __half2float(__ldg(&input[base_j + 2])), sum);
            sum = __fmaf_rn(s_qmap[b1 & 0xF] * scale, __half2float(__ldg(&input[base_j + 3])), sum);

            uint8_t b2 = (packed4 >> 16) & 0xFF;
            sum = __fmaf_rn(s_qmap[b2 >> 4] * scale, __half2float(__ldg(&input[base_j + 4])), sum);
            sum = __fmaf_rn(s_qmap[b2 & 0xF] * scale, __half2float(__ldg(&input[base_j + 5])), sum);

            uint8_t b3 = (packed4 >> 24) & 0xFF;
            sum = __fmaf_rn(s_qmap[b3 >> 4] * scale, __half2float(__ldg(&input[base_j + 6])), sum);
            sum = __fmaf_rn(s_qmap[b3 & 0xF] * scale, __half2float(__ldg(&input[base_j + 7])), sum);
        }

        int rem_start = vec_per_row << 2;
        for (int bi = rem_start + threadIdx.x; bi < bytes_per_row; bi += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + bi]);
            int j0 = bi * 2;
            float sc = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            sum = __fmaf_rn(s_qmap[packed >> 4] * sc, __half2float(__ldg(&input[j0])), sum);
            sum = __fmaf_rn(s_qmap[packed & 0xF] * sc, __half2float(__ldg(&input[j0 + 1])), sum);
        }

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

__device__ __forceinline__ void init_nf4_table(float* s_qmap) {
    if (threadIdx.x < 16) {
        const float T[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = T[threadIdx.x];
    }
    __syncthreads();
}

// ============================================================================
// Kernel wrappers (use Q4L when is_q4l=1, NF4 otherwise)
// ============================================================================

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ a,
    const half* __restrict__ x, half* __restrict__ y,
    int in_dim, int out_dim, int block_size, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int first_row = blockIdx.x * ROWS_PER_BLOCK;
    if (is_q4l)
        q4l_gemv_block(s_sums, w, a, x, y, first_row, out_dim, in_dim);
    else
        nf4_gemv_block(s_qmap, s_sums, w, a, x, y, first_row, out_dim, in_dim);
}

__global__ void nf4_fused_2_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const half* __restrict__ x, int in_dim, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab) {
        if (is_q4l) q4l_gemv_block(s_sums, aw, aa, x, ay, blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay, blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    } else {
        if (is_q4l) q4l_gemv_block(s_sums, bw, ba, x, by, (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by, (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
    }
}

__global__ void nf4_fused_3_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const uint8_t* __restrict__ cw, const float* __restrict__ ca,
    half* __restrict__ cy, int cd,
    const half* __restrict__ x, int in_dim, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int bb = (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int first;
    if (blockIdx.x < ab) {
        first = blockIdx.x * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, aw, aa, x, ay, first, ad, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay, first, ad, in_dim);
    } else if (blockIdx.x < ab + bb) {
        first = (blockIdx.x - ab) * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, bw, ba, x, by, first, bd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by, first, bd, in_dim);
    } else {
        first = (blockIdx.x - ab - bb) * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, cw, ca, x, cy, first, cd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, cw, ca, x, cy, first, cd, in_dim);
    }
}

// ============================================================================
// Launch functions — is_q4l=0 for NF4, is_q4l=1 for Q4 Linear
// ============================================================================

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* p, const float* a, const half* x, half* y,
    int od, int id, int bs, cudaStream_t s
) {
    int nb = (od + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(p, a, x, y, id, od, bs, 0);
}

// Q4 Linear variant (no lookup table)
void launch_q4l_gemv(
    const uint8_t* p, const float* scales, const half* x, half* y,
    int od, int id, int bs, cudaStream_t s
) {
    int nb = (od + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(p, scales, x, y, id, od, bs, 1);
}

void launch_nf4_fused_2(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, x, id, 0);
}

void launch_q4l_fused_2(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, x, id, 1);
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
    nf4_fused_3_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, cw, ca, cy, cd, x, id, 0);
}

void launch_q4l_fused_3(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const uint8_t* cw, const float* ca, half* cy, int cd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (cd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_3_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, cw, ca, cy, cd, x, id, 1);
}

} // extern "C"
