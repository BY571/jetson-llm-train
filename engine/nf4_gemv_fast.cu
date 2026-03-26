/**
 * NF4 GEMV kernels for Jetson Orin (sm_87).
 *
 * Includes single, fused-2, and fused-3 projection variants.
 *
 * Key design: shared memory NF4 table (16 floats) + fp32 accumulation +
 * warp shuffle reduction. Constant memory NF4 tables are 32x slower due
 * to divergent access (each thread reads different nibble value).
 * half2 FMA path also slower due to float→half conversion overhead.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// Shared memory layout: s_qmap[16] + s_sums[ROWS_PER_BLOCK * N_WARPS]
// Total: 16*4 + 4*8*4 = 192 bytes — negligible

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

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row;
             byte_idx += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + byte_idx]);
            int j0 = byte_idx * 2;
            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            float w0 = s_qmap[packed >> 4] * scale;
            float w1 = s_qmap[packed & 0x0F] * scale;
            float x0 = __half2float(__ldg(&input[j0]));
            float x1 = __half2float(__ldg(&input[j0 + 1]));
            sum = __fmaf_rn(w0, x0, sum);
            sum = __fmaf_rn(w1, x1, sum);
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

// Initialize shared NF4 table (called once per block)
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

// Single-projection kernel
__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ a,
    const half* __restrict__ x, half* __restrict__ y,
    int in_dim, int out_dim, int block_size
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    init_nf4_table(s_qmap);
    nf4_gemv_block(s_qmap, s_sums, w, a, x, y,
                   blockIdx.x * ROWS_PER_BLOCK, out_dim, in_dim);
}

// Fused 2-projection kernel
__global__ void nf4_fused_2_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const half* __restrict__ x, int in_dim
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab)
        nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay,
                       blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    else
        nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by,
                       (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
}

// Fused 3-projection kernel
__global__ void nf4_fused_3_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const uint8_t* __restrict__ cw, const float* __restrict__ ca,
    half* __restrict__ cy, int cd,
    const half* __restrict__ x, int in_dim
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int bb = (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab)
        nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay,
                       blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    else if (blockIdx.x < ab + bb)
        nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by,
                       (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
    else
        nf4_gemv_block(s_qmap, s_sums, cw, ca, x, cy,
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
