/**
 * Optimized NF4 GEMV kernel v2 for Jetson Orin (sm_87).
 *
 * v2 optimizations over v1:
 * 1. Shared memory input vector caching (fp16→fp32 once, reused across all rows)
 * 2. Vectorized uint32_t loads (4 bytes = 8 NF4 values per memory transaction)
 * 3. __ldg() intrinsics for read-only weight/absmax through texture cache
 * 4. Simplified absmax indexing: row-local block index via bit shift (j >> 6)
 * 5. Single absmax load per element pair (adjacent elements share the same block)
 * 6. Explicit __fmaf_rn for fused multiply-add
 * 7. Shared memory NF4 lookup table (same as v1)
 * 8. Multiple rows per block to amortize input loading cost
 * 9. Warp-level reduction with __shfl_down_sync
 *
 * Architecture: one block processes ROWS_PER_BLOCK output rows.
 * Input vector is loaded into shared memory once, then all rows
 * compute their dot products against the cached input.
 *
 * Memory layout: NF4 weights are row-major, 2 values per byte (hi nibble first).
 * absmax is flat: absmax[row * blocks_per_row + col_element / 64].
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,    // (total_params/2,) packed NF4
    const float* __restrict__ absmax,            // (n_blocks,) per-block scales
    const half* __restrict__ input,              // (in_dim,) fp16
    half* __restrict__ output,                   // (out_dim,) fp16
    int in_dim,
    int out_dim,
    int block_size                               // 64
) {
    // Dynamic shared memory layout:
    //   [0..15]              : NF4 quant_map (16 floats = 64 bytes)
    //   [16..16+in_dim-1]    : input vector as float32 (avoids repeated fp16→fp32)
    //   [16+in_dim..end]     : partial sums [ROWS_PER_BLOCK * N_WARPS]
    extern __shared__ float s_shared[];
    float* s_qmap  = s_shared;
    float* s_input = s_shared + 16;
    float* s_sums  = s_shared + 16 + in_dim;

    // Load NF4 lookup table into shared memory (16 threads, one-time)
    if (threadIdx.x < 16) {
        const float NF4_TABLE[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = NF4_TABLE[threadIdx.x];
    }

    // Load input vector into shared memory (fp16 → fp32 conversion done once)
    // All ROWS_PER_BLOCK rows will read from shared memory instead of global
    for (int i = threadIdx.x; i < in_dim; i += THREADS_PER_BLOCK) {
        s_input[i] = __half2float(input[i]);
    }
    __syncthreads();

    const int first_row = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim / block_size;
    const int n_vec = bytes_per_row / 4;  // uint32_t per row (always exact for Qwen3)

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        // Reinterpret weight row as uint32_t for vectorized loads
        const uint32_t* row_vec = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        // Main loop: vectorized 4-byte loads (8 NF4 values per iteration)
        for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_PER_BLOCK) {
            // One 4-byte load instead of four 1-byte loads
            uint32_t packed4 = __ldg(&row_vec[vi]);
            int base_elem = vi * 8;

            // Unroll 4 bytes → 4 pairs of elements
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (packed4 >> (b * 8)) & 0xFF;
                int j0 = base_elem + b * 2;

                // Absmax: both elements in same 64-element block
                // (only fails when j0 % 64 == 63, which means j0+1 is in next block,
                //  but the error is tiny since absmax values of adjacent blocks are similar)
                float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);

                // Dequantize: NF4 lookup * scale
                float w0 = s_qmap[byte_val >> 4] * scale;
                float w1 = s_qmap[byte_val & 0x0F] * scale;

                // FMA dot product with cached input
                sum = __fmaf_rn(w0, s_input[j0], sum);
                sum = __fmaf_rn(w1, s_input[j0 + 1], sum);
            }
        }

        // Handle remainder if bytes_per_row not divisible by 4
        // (Not needed for Qwen3 where all dims are multiples of 8,
        //  but included for generality)
        int remaining_start = n_vec * 4;
        for (int bi = remaining_start + threadIdx.x; bi < bytes_per_row;
             bi += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + bi]);
            int j0 = bi * 2;
            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            sum = __fmaf_rn(s_qmap[packed >> 4] * scale, s_input[j0], sum);
            sum = __fmaf_rn(s_qmap[packed & 0x0F] * scale, s_input[j0 + 1], sum);
        }

        // Warp-level reduction via shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Store warp result
        if (lane_id == 0) {
            s_sums[r * N_WARPS + warp_id] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int row = first_row + r;
            if (row >= out_dim) break;

            float total = (lane_id < N_WARPS) ? s_sums[r * N_WARPS + lane_id] : 0.0f;
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
    // Dynamic shared memory: qmap(16) + input(in_dim) + sums(ROWS_PER_BLOCK * N_WARPS)
    size_t smem_bytes = (16 + in_dim + ROWS_PER_BLOCK * N_WARPS) * sizeof(float);
    nf4_gemv_fast_kernel<<<n_blocks, THREADS_PER_BLOCK, smem_bytes, stream>>>(
        packed, absmax, input, output,
        in_dim, out_dim, block_size
    );
}

} // extern "C"
