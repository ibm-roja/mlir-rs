#!/bin/bash

# Install dependencies
sudo apt-get update && \
    apt-get install -y \
    wget build-essential curl zlib1g-dev libzstd-dev pkg-config \
    libssl-dev cmake zlib1g-dev libcurl4-openssl-dev libelf-dev \
    libdw-dev binutils-dev libiberty-dev lsb-release wget software-properties-common gnupg xz-utils \
    libncurses-dev libxml2-dev ninja-build clang lld git

sudo wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt update && \
    apt install -y cmake

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

# Install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . $HOME/.cargo/env 

PATH=${PATH}:/usr/lib/llvm-${LLVM_VERSION}/bin:/root/.cargo/bin
export PATH
export RUSTUP_HOME="/root/.rustup"
export CARGO_HOME="/root/.cargo"
export CC="clang"
export CXX="clang++"