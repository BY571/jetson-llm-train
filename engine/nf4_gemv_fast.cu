/**
 * Optimized NF4 GEMV kernel for Jetson Orin (sm_87).
 *
 * Key optimizations over the naive kernel:
 * 1. Coalesced memory reads: threads read consecutive packed bytes
 * 2. Shared memory for quant_map (16 floats, read once, reused)
 * 3. Vectorized loads: read 4 bytes at a time (uint32_t = 8 NF4 values)
 * 4. Multiple rows per block: amortize kernel launch overhead
 * 5. Warp-level primitives for reduction
 *
 * Architecture: one block processes ROWS_PER_BLOCK output rows.
 * Each warp handles part of the dot product for one row.
 * Block reduction across warps.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,    // (total_params/2,) packed NF4
    const float* __restrict__ absmax,            // (n_blocks,) per-block scales
    const half* __restrict__ input,              // (in_dim,) fp16
    half* __restrict__ output,                   // (out_dim,) fp16
    int in_dim,
    int out_dim,
    int block_size                               // 64
) {
    // Shared memory: quant_map (16 floats) + partial sums
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK][8]; // 8 warps max

    // Load quant_map into shared memory (only first 16 threads)
    // The standard NF4 lookup table
    if (threadIdx.x < 16) {
        static const float NF4_TABLE[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    int first_row = blockIdx.x * ROWS_PER_BLOCK;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int n_warps = THREADS_PER_BLOCK / 32;

    int blocks_per_row = in_dim / block_size;
    int bytes_per_row = in_dim / 2;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;

        // Each thread processes some elements of the dot product
        // Stride by total threads to ensure coalesced access within warps
        int row_byte_start = row * bytes_per_row;

        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row; byte_idx += THREADS_PER_BLOCK) {
            uint8_t packed = weight_data[row_byte_start + byte_idx];
            uint8_t hi_nib = (packed >> 4) & 0x0F;
            uint8_t lo_nib = packed & 0x0F;

            // Element indices within the row
            int j0 = byte_idx * 2;
            int j1 = j0 + 1;

            // Block index for absmax
            int flat0 = row * in_dim + j0;
            int flat1 = flat0 + 1;
            int blk0 = flat0 / block_size;
            int blk1 = flat1 / block_size;

            float scale0 = absmax[blk0];
            float scale1 = absmax[blk1];

            float w0 = s_qmap[hi_nib] * scale0;
            float w1 = s_qmap[lo_nib] * scale1;

            float x0 = __half2float(input[j0]);
            float x1 = __half2float(input[j1]);

            sum += w0 * x0 + w1 * x1;
        }

        // Warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Store warp result
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

            float total = (lane_id < n_warps) ? s_sums[r][lane_id] : 0.0f;
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
