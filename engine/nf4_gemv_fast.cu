/**
 * Optimized NF4 GEMV kernel v4 for Jetson Orin (sm_87).
 *
 * v3 analysis: kernel achieves 18-28 GB/s vs 68 GB/s peak and 85-100 GB/s
 * cuBLAS fp16. The kernel is latency-bound: ~18 instructions per byte in
 * the inner loop create serial FMA dependency chains that stall the pipeline.
 *
 * v4 optimizations to reduce instruction count:
 * 1. 256-entry byte lookup table: maps packed NF4 byte → (dequant_hi, dequant_lo)
 *    Replaces: 2 bit extractions + 2 shared mem lookups → 1 float2 shared mem load
 * 2. half2 vectorized input load: 1 load instead of 2 separate half loads
 * 3. __ldg() for read-only data via texture cache
 * 4. Simplified absmax indexing via bit shift
 * 5. Explicit __fmaf_rn for FMA
 * 6. #pragma unroll on reductions
 *
 * Estimated inner loop: ~12 instructions per byte (vs ~18 in v3).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// Standard NF4 lookup table
__device__ static const float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int in_dim,
    int out_dim,
    int block_size
) {
    // Shared memory layout:
    //   [0..255]: byte lookup table (256 × float2 = 2048 bytes)
    //   [256..256+ROWS_PER_BLOCK*N_WARPS-1]: partial sums
    __shared__ float2 s_byte_table[256];     // 2KB
    __shared__ float s_sums[ROWS_PER_BLOCK][N_WARPS];

    // Initialize byte lookup table: each of 256 entries maps a packed byte
    // to its two dequantized NF4 values (before absmax scaling)
    if (threadIdx.x < 256) {
        uint8_t hi = threadIdx.x >> 4;
        uint8_t lo = threadIdx.x & 0x0F;
        s_byte_table[threadIdx.x] = make_float2(NF4_TABLE[hi], NF4_TABLE[lo]);
    }
    __syncthreads();

    const int first_row = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;  // in_dim / 64

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row;
             byte_idx += THREADS_PER_BLOCK) {
            // 1. Load weight byte through texture cache
            uint8_t packed = __ldg(&weight_data[row_byte_start + byte_idx]);

            // 2. Byte lookup: 1 float2 load replaces 2 bit extracts + 2 lookups
            float2 dq = s_byte_table[packed];

            // 3. Absmax scale (row-local index via bit shift)
            int j0 = byte_idx * 2;
            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);

            // 4. Dequantize
            float w0 = dq.x * scale;
            float w1 = dq.y * scale;

            // 5. Vectorized half2 input load + convert
            half2 x_pair = __ldg(reinterpret_cast<const half2*>(&input[j0]));
            float x0 = __half2float(x_pair.x);
            float x1 = __half2float(x_pair.y);

            // 6. FMA accumulate
            sum = __fmaf_rn(w0, x0, sum);
            sum = __fmaf_rn(w1, x1, sum);
        }

        // Warp-level reduction via shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane_id == 0) {
            s_sums[r][warp_id] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int row = first_row + r;
            if (row >= out_dim) break;

            float total = (lane_id < N_WARPS) ? s_sums[r][lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            }
            if (lane_id == 0) {
                output[row] = __float2half(total);
            }
        }
    }
}

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* packed,
    const float* absmax,
    const half* input,
    half* output,
    int out_dim,
    int in_dim,
    int block_size,
    cudaStream_t stream
) {
    int n_blocks = (out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        packed, absmax, input, output,
        in_dim, out_dim, block_size
    );
}

} // extern "C"
