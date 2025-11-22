#!/bin/bash
# Script para descargar LibTorch automáticamente
# Uso: ./download_libtorch.sh [cuda_version]
# Ejemplos:
#   ./download_libtorch.sh cpu       # CPU-only
#   ./download_libtorch.sh cu118     # CUDA 11.8
#   ./download_libtorch.sh cu121     # CUDA 12.1

set -e

CUDA_VERSION=${1:-cpu}
LIBTORCH_VERSION="2.1.0"
INSTALL_DIR="./libtorch"

echo "==================================="
echo "LibTorch Downloader"
echo "==================================="
echo "CUDA Version: $CUDA_VERSION"
echo "LibTorch Version: $LIBTORCH_VERSION"
echo "Install Directory: $INSTALL_DIR"
echo ""

# Determinar la URL según la plataforma y versión de CUDA
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ "$CUDA_VERSION" == "cpu" ]; then
        URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
    elif [ "$CUDA_VERSION" == "cu118" ]; then
        URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu118.zip"
    elif [ "$CUDA_VERSION" == "cu121" ]; then
        URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu121.zip"
    else
        echo "Error: CUDA version not supported. Use: cpu, cu118, or cu121"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    # macOS solo soporta CPU
    URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-${LIBTORCH_VERSION}.zip"
else
    echo "Error: OS not supported. Only Linux and macOS are supported by this script."
    echo "For Windows, download manually from https://pytorch.org"
    exit 1
fi

echo "Download URL: $URL"
echo ""

# Descargar
echo "Downloading LibTorch..."
wget -O libtorch.zip "$URL"

# Descomprimir
echo "Extracting..."
unzip -q libtorch.zip
rm libtorch.zip

echo ""
echo "==================================="
echo "✅ LibTorch installed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Build the project:"
echo "     mkdir build && cd build"
echo "     cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch .."
echo "     make -j\$(nproc)"
echo ""
echo "  2. Run the DQN agent:"
echo "     ./dqn"
echo ""
