#!/bin/bash
# Build llama.cpp for Jetson Orin (CUDA sm_87)
# Run inside the Docker container

set -e

echo "=== Building llama.cpp for Jetson Orin ==="

cd /workspace

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp

# Build with CUDA for Jetson Orin (sm_87 = Ampere)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

echo ""
echo "=== llama.cpp built successfully ==="
echo "  Server: $(pwd)/build/bin/llama-server"
echo "  Convert: $(pwd)/convert_hf_to_gguf.py"
echo ""
echo "Next steps:"
echo "  1. Convert model: python3 convert_hf_to_gguf.py <model_dir> --outfile model.gguf --outtype q4_k_m"
echo "  2. Start server:  ./build/bin/llama-server -m model.gguf -ngl 999 -c 2048 --flash-attn"
echo "  3. Run training:  python3 improvements/01_llamacpp_generation.py --mode train"
