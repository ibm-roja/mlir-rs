#!/bin/bash

# Set LLVM version
if [ -z "$LLVM_VERSION_FULL" ]; then
    LLVM_VERSION_FULL="18.1.6"
fi

# Set installation directory
INSTALL_DIR="/usr/lib/llvm-$LLVM_VERSION"

# Function to download and extract LLVM
download_and_extract() {
    URL=$1
    echo "Downloading LLVM from $URL..."
    wget -O llvm.tar.xz "$URL"
    echo "Extracting LLVM to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    tar -xJf llvm.tar.xz -C "$INSTALL_DIR" --strip-components=1
    rm llvm.tar.xz
}

# Install natively since llvm-18.X does not provide the proper linux x86 binaries
install_llvm() {
    apt remove -y *llvm*
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh ${LLVM_FULL_VERSION} all
    apt-get install -y libmlir-18-dev mlir-18-tools
    rm -f ./llvm.sh
}

# Detect architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        # Linux x86_64
        install_llvm
        ;;
    arm64|aarch64)
        # macOS arm64 (assuming running within Docker and thus using Linux binaries)
        URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION_FULL/clang+llvm-$LLVM_VERSION_FULL-aarch64-linux-gnu.tar.xz"
        download_and_extract "$URL"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac