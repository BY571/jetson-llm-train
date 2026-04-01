#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>

// Maximum number of transformer layers (covers all Qwen3 variants)
constexpr int MAX_LAYERS = 128;

// Layer type for hybrid architectures (Qwen3.5 = transformer + Mamba-2)
enum LayerType : int {
    LAYER_ATTENTION = 0,   // standard GQA attention with KV cache
    LAYER_SSM = 1,         // Gated Delta Rule SSM (Mamba-2 style)
};

// Runtime model configuration — loaded from config.json alongside weights.
// Replaces the old compile-time qwen3:: namespace.
struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;          // Q heads (attention layers)
    int num_kv_heads;       // KV heads (attention layers, GQA)
    int head_dim;           // head dimension (attention layers)
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;
    int rope_dim;              // rotary dim (default = head_dim, partial_rotary_factor < 1 → smaller)

    // Hybrid architecture: SSM (Gated Delta Rule) layer config
    // Zero if model is pure transformer (e.g., Qwen3-0.6B)
    int ssm_num_k_heads;    // SSM key heads (16 for Qwen3.5-0.8B)
    int ssm_num_v_heads;    // SSM value heads (32 for Qwen3.5-0.8B)
    int ssm_k_head_dim;     // key head dim (128)
    int ssm_v_head_dim;     // value head dim (128)
    int ssm_conv_kernel;    // causal conv1d kernel size (4)

    // Per-layer type: 0=attention, 1=SSM
    LayerType layer_type[MAX_LAYERS];

    bool gated_attn;           // Qwen3.5: q_proj outputs 2x (query + gate), gate applied after attention

    // Derived dimensions
    int q_dim() const { return num_heads * head_dim; }  // attention Q dimension
    int q_proj_dim() const { return num_heads * head_dim * (gated_attn ? 2 : 1); }  // q_proj output dimension
    int kv_dim() const { return num_kv_heads * head_dim; }
    int gqa_groups() const { return num_heads / num_kv_heads; }

    // SSM derived dimensions
    int ssm_key_dim() const { return ssm_num_k_heads * ssm_k_head_dim; }
    int ssm_value_dim() const { return ssm_num_v_heads * ssm_v_head_dim; }
    int ssm_conv_dim() const { return 2 * ssm_key_dim() + ssm_value_dim(); }
    int ssm_state_dim() const { return ssm_k_head_dim * ssm_v_head_dim; } // per head
    bool is_hybrid() const { return ssm_num_v_heads > 0; }

    // Count layer types
    int num_attn_layers() const {
        int n = 0;
        for (int i = 0; i < num_layers; i++) if (layer_type[i] == LAYER_ATTENTION) n++;
        return n;
    }
    int num_ssm_layers() const { return num_layers - num_attn_layers(); }

    // Built-in presets
    static ModelConfig qwen3_0_6b() {
        ModelConfig c = {};
        c.hidden_size = 1024; c.intermediate_size = 3072; c.num_layers = 28;
        c.num_heads = 16; c.num_kv_heads = 8; c.head_dim = 128;
        c.vocab_size = 151936; c.rms_norm_eps = 1e-6f; c.rope_theta = 1000000.0f;
        c.rope_dim = 128; c.gated_attn = false;
        for (int i = 0; i < 28; i++) c.layer_type[i] = LAYER_ATTENTION;
        return c;
    }
    static ModelConfig qwen3_1_7b() {
        ModelConfig c = {};
        c.hidden_size = 2048; c.intermediate_size = 6144; c.num_layers = 28;
        c.num_heads = 16; c.num_kv_heads = 8; c.head_dim = 128;
        c.vocab_size = 151936; c.rms_norm_eps = 1e-6f; c.rope_theta = 1000000.0f;
        c.rope_dim = 128; c.gated_attn = false;
        for (int i = 0; i < 28; i++) c.layer_type[i] = LAYER_ATTENTION;
        return c;
    }

    // Load from JSON file (written by convert_weights.py)
    static ModelConfig from_json(const std::string& path);
};

// 4-bit quantized weight: used for both NF4 (non-linear, lookup table) and Q4L
// (linear, (nibble - 8) * scale) formats. The format is determined by the
// is_q4l flags on TransformerLayerWeights and ModelWeights, not by this struct.
// For NF4: absmax stores per-block absmax values, dequant uses quant_map lookup.
// For Q4L: absmax stores per-block scale factors, dequant is linear arithmetic.
struct NF4Weight {
    uint8_t* data;          // packed NF4 values (2 per byte), on GPU
    float* absmax;          // per-block scale factors (float32, pre-dequantized), on GPU
    float* quant_map;       // NF4 dequant lookup table (16 entries), on GPU
    half* fp16_cache;       // pre-dequanted fp16 weights (nullptr = not cached)
    int out_dim;            // output dimension (rows)
    int in_dim;             // input dimension (cols)
    int block_size;         // typically 64
    int n_blocks;           // out_dim * in_dim / block_size

    int total_params() const { return out_dim * in_dim; }
    size_t data_bytes() const { return (size_t)total_params() / 2; }
};

// LoRA adapter weights for one linear layer
struct LoRAAdapter {
    half* A;    // (rank, in_features) - stored row-major
    half* B;    // (out_features, rank) - stored row-major
    int rank;
    int in_features;
    int out_features;
    float scale; // alpha / rank
};

// One transformer layer's weights
struct TransformerLayerWeights {
    // Attention: fp16 for layer 0, NF4 for layers 1-27 in unsloth model
    // Only one of fp16 or nf4 is non-null per projection
    half* q_proj_fp16;      // (Q_DIM, HIDDEN) if fp16
    half* k_proj_fp16;      // (KV_DIM, HIDDEN) if fp16
    half* v_proj_fp16;      // (KV_DIM, HIDDEN) if fp16
    half* o_proj_fp16;      // (HIDDEN, Q_DIM) if fp16

    NF4Weight q_proj_nf4;   // if NF4
    NF4Weight k_proj_nf4;   // if NF4
    NF4Weight v_proj_nf4;   // if NF4
    NF4Weight o_proj_nf4;   // if NF4
    bool attn_is_nf4 = false;
    bool attn_is_q4l = false;  // Q4 linear (no lookup table)

    // MLP: either fp16 (dequantized) or NF4 (native quantized)
    // Only one of these is non-null per layer
    half* gate_proj_fp16;   // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* up_proj_fp16;     // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* down_proj_fp16;   // (HIDDEN, INTERMEDIATE) if fp16 mode

    NF4Weight gate_proj_nf4;  // if NF4 mode
    NF4Weight up_proj_nf4;    // if NF4 mode
    NF4Weight down_proj_nf4;  // if NF4 mode
    bool mlp_is_nf4 = false;
    bool mlp_is_q4l = false;

    // Norms (fp16, small)
    half* input_layernorm;  // (HIDDEN,)
    half* post_attn_layernorm; // (HIDDEN,)

    // QKNorm (Qwen3 specific — RMSNorm applied to Q and K after projection)
    half* q_norm;           // (HEAD_DIM,) = (128,)
    half* k_norm;           // (HEAD_DIM,) = (128,)

    // Optional LoRA adapters (nullptr if not used)
    LoRAAdapter* lora_q;
    LoRAAdapter* lora_k;
    LoRAAdapter* lora_v;
    LoRAAdapter* lora_o;
    LoRAAdapter* lora_gate;
    LoRAAdapter* lora_up;
    LoRAAdapter* lora_down;

    // === SSM (Gated Delta Rule) weights — used when layer_type == LAYER_SSM ===
    // Large projections (Q4L quantized or fp16)
    half* ssm_in_proj_qkv_fp16;    // (conv_dim, hidden) — Q,K,V fused
    NF4Weight ssm_in_proj_qkv_nf4;
    half* ssm_in_proj_z_fp16;      // (value_dim, hidden) — gate
    NF4Weight ssm_in_proj_z_nf4;
    half* ssm_out_proj_fp16;       // (hidden, value_dim) — output
    NF4Weight ssm_out_proj_nf4;
    bool ssm_is_q4l = false;

    // Small projections (always fp16, too small for quantization)
    half* ssm_in_proj_a_fp16;      // (num_v_heads, hidden) — decay input
    half* ssm_in_proj_b_fp16;      // (num_v_heads, hidden) — beta/gate input

    // Conv1d (fp16)
    half* ssm_conv1d_weight;       // (conv_dim, conv_kernel) — depthwise causal conv
    half* ssm_conv1d_bias;         // (conv_dim,) or nullptr

    // SSM parameters (fp16, tiny)
    half* ssm_A_log;               // (num_v_heads,) — log eigenvalues
    half* ssm_dt_bias;             // (num_v_heads,) — dt bias

    // Gated RMSNorm (fp16)
    half* ssm_norm_weight;         // (v_head_dim,) — shared across heads

    // LoRA for SSM projections
    LoRAAdapter* lora_ssm_qkv;
    LoRAAdapter* lora_ssm_z;
    LoRAAdapter* lora_ssm_out;
};

// KV cache for one layer
struct KVCache {
    // FP16 mode (kv_bits == 0)
    half* key;   // (max_seq_len, KV_DIM)
    half* value; // (max_seq_len, KV_DIM)
    // TurboQuant mode (kv_bits == 2): 2-bit packed quantized storage
    uint8_t* key_quant;    // (max_seq_len, KV_DIM / 4) — 4 values per byte
    uint8_t* value_quant;  // same
    half* key_norms;       // (max_seq_len, num_kv_heads) per-head norms
    half* value_norms;     // same
};

// Full model weights
struct ModelWeights {
    // Embedding (fp16, used for both token lookup and LM head via cuBLAS)
    half* embed_tokens;     // (VOCAB_SIZE, HIDDEN)

    // Transformer layers (dynamic count, up to MAX_LAYERS)
    TransformerLayerWeights layers[MAX_LAYERS];

    // Final norm
    half* final_layernorm;  // (HIDDEN,)

    // Model-wide quantization format (set during loading)
    bool is_q4l = false;  // true if weights are Q4 Linear (not NF4)
};

// Inference state (pre-allocated buffers)
struct InferenceState {
    // KV caches for all layers (dynamic count, up to MAX_LAYERS)
    KVCache kv_cache[MAX_LAYERS];
    int max_seq_len;
    int current_pos;    // current position in KV cache

    // Scratch buffers (reused across layers)
    half* hidden;       // (HIDDEN,) current hidden state
    half* residual;     // (HIDDEN,) residual connection
    half* q_buf;        // (Q_DIM,)
    half* k_buf;        // (KV_DIM,)
    half* v_buf;        // (KV_DIM,)
    half* attn_out;     // (Q_DIM,)
    half* gate_buf;     // (INTERMEDIATE,)
    half* up_buf;       // (INTERMEDIATE,)
    half* ffn_out;      // (HIDDEN,)
    float* logits;      // (VOCAB_SIZE,) float32 for numerical stability in sampling

    // Attention scratch
    float* attn_scores; // (NUM_HEADS, max_seq_len) float32 for softmax stability

    // RoPE precomputed cos/sin
    half* rope_cos;     // (max_seq_len, HEAD_DIM/2)
    half* rope_sin;     // (max_seq_len, HEAD_DIM/2)

    // GPU-side sampling result
    int* sample_result; // single int on GPU

    // dp4a input quantization buffers (for Q4L dp4a GEMV)
    int8_t* q8_data;        // (INTERMEDIATE_SIZE,) quantized input
    float* q8_scales;       // (INTERMEDIATE_SIZE/64,) per-block scale
    float* q8_sums;         // (INTERMEDIATE_SIZE/64,) per-block sum of q8 values

    // LoRA scratch buffer (for A @ x intermediate, max rank = 64)
    half* lora_scratch; // (max_lora_rank,)

    // TurboQuant: 2-bit KV cache quantization (Zandieh et al. 2025)
    half* turbo_rotation;   // (HEAD_DIM, HEAD_DIM) random orthogonal matrix, shared across layers
};

// Simple GPU arena: one cudaMalloc, bump-pointer suballocation.
// Prevents fragmentation with PyTorch's caching allocator.
struct GpuArena {
    char* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;

    // Allocate from arena (256-byte aligned for cuBLAS)
    void* alloc(size_t bytes) {
        offset = (offset + 255) & ~(size_t)255;
        void* ptr = base + offset;
        offset += bytes;
        return ptr;
    }

    void reset() { offset = 0; }
    size_t used() const { return offset; }
};

// Batched generation state (G sequences in parallel)
struct BatchState {
    int G;              // batch size (number of sequences)
    int max_seq_len;

    // Activation buffers: (dim, G) column-major for cuBLAS
    half* hidden;       // (HIDDEN_SIZE, G)
    half* residual;     // (HIDDEN_SIZE, G)
    half* norm_buf;     // (HIDDEN_SIZE, G) temp for norm output
    half* q_buf;        // (Q_PROJ_DIM, G) — full q_proj output; for gated attn, first half = Q, second half = gate
    half* k_buf;        // (KV_DIM, G)
    half* v_buf;        // (KV_DIM, G)
    half* attn_out;     // (Q_DIM, G)
    half* gate_buf;     // (INTERMEDIATE_SIZE, G)
    half* up_buf;       // (INTERMEDIATE_SIZE, G)
    float* logits;      // (VOCAB_SIZE, G) fp32

    // Attention scratch
    float* attn_scores; // (G * NUM_HEADS * max_seq_len)

    // Batched KV cache: (G * max_seq_len * KV_DIM) per layer
    half* kv_keys[MAX_LAYERS];
    half* kv_values[MAX_LAYERS];

    // TurboQuant 2-bit KV cache (used when kv_bits == 2)
    uint8_t* kv_keys_q[MAX_LAYERS];    // (G * max_seq_len, KV_DIM / 4)
    uint8_t* kv_values_q[MAX_LAYERS];  // same
    half* kv_keys_norms[MAX_LAYERS];   // (G * max_seq_len, num_kv_heads)
    half* kv_values_norms[MAX_LAYERS]; // same

    // SSM recurrent state (Gated Delta Rule, for LAYER_SSM layers)
    float* ssm_state[MAX_LAYERS];      // (G, num_v_heads, k_head_dim, v_head_dim) per SSM layer — fp32 to avoid overflow
    float* ssm_conv_state[MAX_LAYERS]; // (G, conv_dim, conv_kernel-1) per SSM layer — fp32

    // SSM scratch buffers (reused across layers, sized for largest SSM dims)
    float* ssm_qkv_fp32;              // (conv_dim, G) — QKV after conv1d+SiLU (fp32 for SSM precision)
    half* ssm_qkv_buf;                // (conv_dim, G) — QKV projection output (fp16 from GEMM)
    half* ssm_z_buf;                   // (value_dim, G) — gate projection
    float* ssm_y_fp32;                // (value_dim, G) — SSM delta rule output (fp32, avoids overflow)
    half* ssm_y_buf;                   // (value_dim, G) — after gated rmsnorm (fp16, for out_proj GEMM)
    float* ssm_dt_buf;                // (num_v_heads, G) — decay values
    half* ssm_a_buf;                   // (num_v_heads, G) — in_proj_a output
    half* ssm_b_buf;                   // (num_v_heads, G) — in_proj_b output (beta input)

    // SSM chunked prefill workspace (reused across layers)
    float* ssm_chunk_Q;               // (H, T_padded, D) — rearranged Q
    float* ssm_chunk_K;               // (H, T_padded, D) — rearranged K
    float* ssm_chunk_V;               // (H, T_padded, D) — rearranged V
    float* ssm_chunk_K_beta;          // (H, T_padded, D) — K * beta
    float* ssm_chunk_V_beta;          // (H, T_padded, D) — V * beta
    float* ssm_chunk_g;               // (H, T) — raw gate values
    float* ssm_chunk_beta;            // (H, T) — beta values
    float* ssm_chunk_output;          // (H, T_padded, D) — chunked output
    float* ssm_chunk_workspace;       // (H, ws_per_head) — per-chunk scratch

    // Q4L dequant scratch (largest projection = INTERMEDIATE_SIZE * HIDDEN_SIZE)
    half* dequant_scratch;

    // LoRA scratch for batched A @ x intermediate (max_rank * G)
    half* lora_scratch;

    // Per-sequence state
    int* h_positions;   // host (G,)
    int* d_positions;   // device (G,)
    int* h_tokens;      // host (G,) sampled tokens
    int* d_tokens;      // device (G,)
    bool* h_finished;   // host (G,)
    float* h_randoms;   // host (G,)
    float* d_randoms;   // device (G,)

    // Device-side token history (for amortized stop checking)
    int* d_token_history;   // device (max_new_tokens * G)
    int* h_token_history;   // host (max_new_tokens * G) -- bulk read on sync
    int token_history_size; // allocated size

    // Pre-generated random values for all tokens (max_new_tokens * G)
    float* d_all_randoms;   // device (max_tokens * G)
    int all_randoms_size;   // number of floats allocated

    // Pointer indirection for CUDA graph-compatible sampling
    // d_randoms_ptr is a device-side pointer TO d_all_randoms + step*G.
    // Updated via 8-byte cudaMemcpyAsync before each graph launch.
    // The graph's sampling kernel reads *d_randoms_ptr (stable address).
    float** d_randoms_ptr;  // device: points to current step's randoms slice
};

// Top-level engine
class InferenceEngine {
public:
    InferenceEngine(int max_seq_len = 1024, int kv_bits = 0);
    ~InferenceEngine();

    int kv_bits() const { return kv_bits_; }

    // Load weights from safetensors directory (HF format)
    void load_weights(const std::string& model_dir);

    // Load LoRA adapter from file
    void load_lora(const std::string& lora_dir, float scale = 1.0f);

    // Update a single LoRA adapter from raw fp16 data (for live sync from PyTorch)
    void update_lora_weight(int layer_idx, const char* proj_name,
                            const half* A_data, int A_rows, int A_cols,
                            const half* B_data, int B_rows, int B_cols,
                            float scale);

    // Reset KV cache for new generation
    void reset();

    // Share embedding from PyTorch (avoids 311MB duplicate on Jetson)
    // Pass the raw GPU pointer from model.embed_tokens.weight.data_ptr()
    void share_embedding(void* external_embed_ptr);
    void share_weight(int layer, const char* name, half* ptr);
    void load_config(const std::string& config_path);
    void load_weights_gguf(const std::string& gguf_path);
    bool embed_is_external_ = false;

    // Prefill: process multiple tokens at once
    // Returns logits for the last token
    void prefill(const int* token_ids, int n_tokens);

    // Decode: process one token, returns logits
    void decode(int token_id);

    // Get logits (after prefill or decode)
    float* get_logits() const { return state_.logits; }

    // Fast greedy sampling on GPU (only copies 4 bytes instead of 600KB)
    int sample_greedy_gpu();

    // GPU sampling with temperature + top-p (only copies 4 bytes)
    int sample_gpu(float temperature = 1.0f, float top_p = 0.9f);

    // Full generation: prefill + decode loop
    std::vector<int> generate(const std::vector<int>& prompt,
                               int max_new_tokens = 512,
                               float temperature = 1.0f,
                               float top_p = 0.9f,
                               int eos_token_id = -1,
                               const std::vector<int>& stop_token_ids = {});

    // Batched generation (G sequences in parallel, GEMM with tensor cores)
    std::vector<std::vector<int>> generate_batch(
        const std::vector<std::vector<int>>& prompts,
        int max_new_tokens = 512,
        float temperature = 1.0f,
        float top_p = 0.9f,
        int eos_token_id = -1,
        const std::vector<int>& stop_token_ids = {});

    // Pre-dequant Q4L weights to fp16 for fast batched GEMM
    void cache_weights();

    // Set external arena memory (allocated by PyTorch, avoids allocator contention)
    void set_arena(void* ptr, size_t size);

    // Sleep/wake: free GPU buffers during training, re-allocate for generation
    void sleep();
    void wake();

    const ModelConfig& config() const { return config_; }

private:
    bool weights_cached_ = false;
    int kv_bits_ = 0;  // 0 = fp16 KV cache, 2 = TurboQuant 2-bit
    GpuArena batch_arena_;

    // Dedicated stream for batched decode (isolated from PyTorch's default stream)
    cudaStream_t engine_stream_ = nullptr;

    // CUDA graph for batched decode (prefill uses chunked forward, no graph needed)
    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    int graph_G_ = 0;
    ModelConfig config_;
    ModelWeights weights_;
    InferenceState state_;

    // Forward pass for one token through one layer (dispatches attn vs SSM)
    void forward_layer(int layer_idx);
    void forward_layer_ssm(int layer_idx);  // Gated Delta Rule SSM

    // Allocate GPU buffers (called from load_weights after config is known)
    void allocate_buffers();

    // Precompute RoPE cos/sin tables
    void precompute_rope();

    // Batch generation internals
    BatchState* batch_;  // allocated on first generate_batch call
    void alloc_batch(int G, int max_seq_len);
    void decode_batch(int G, cudaStream_t stream = 0);
    void prefill_chunked(int T, int G, cudaStream_t stream);
    void forward_layer_batch(int layer_idx, int G, cudaStream_t stream);
    void forward_layer_ssm_batch(int layer_idx, int G, cudaStream_t stream);
    void forward_layer_ssm_prefill(int layer_idx, int T, cudaStream_t stream);

    // Batched GEMM: (M,K) @ (K,N) -> (M,N) where N=G
    void batch_gemm(half* out, const half* weight, const half* in,
                    int M, int N, int K, cudaStream_t stream);
    // Q4L batch: dequant to scratch then cuBLAS GEMM
    void batch_gemm_q4l(half* out, const NF4Weight& w, const half* in,
                        int N, cudaStream_t stream);

};
