FROM ubuntu:22.04
ENV LLVM_VERSION=18
ENV LLVM_VERSION_FULL=18.1.4

RUN apt-get update && \
    apt-get install -y \
    wget build-essential curl zlib1g-dev libzstd-dev pkg-config \
    libssl-dev cmake zlib1g-dev libcurl4-openssl-dev libelf-dev \
    libdw-dev binutils-dev libiberty-dev lsb-release wget software-properties-common gnupg xz-utils \
    libncurses-dev libxml2-dev ninja-build clang lld git valgrind

RUN which valgrind || (echo "valgrind not found" && exit 1)

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt update && \
    apt install -y cmake

COPY scripts/* /scripts/
RUN chmod +x /scripts/* && /scripts/llvm-install.sh

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup default nightly-2024-05-02 && \
    cargo install cargo-tarpaulin && \
    rustup component add clippy && \
    rustup component add rustfmt && \
    # Install tools for code coverage
    cargo install cargo-llvm-cov cargo-nextest && \
    rustup component add llvm-tools-preview

ENV PATH=${PATH}:/usr/lib/llvm-${LLVM_VERSION}/bin:/root/.cargo/bin
# Github actions override the default home directory, this is a workaround to find the cargo and rustup directories
ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo

ENV CC=clang \
    CXX=clang++