UNAME_S := $(shell uname -s)

LLVM_VERSION := 18

ifeq ($(UNAME_S),Linux)
    # Attempt to find the specified LLVM version in the default Linux directory
    LLVM_PREFIX := $(shell ls -d /usr/lib/llvm-$(LLVM_VERSION) || ls -d /usr/lib/llvm-* | tail -1)
endif

ifeq ($(UNAME_S),Darwin)
    export LIBRARY_PATH=/opt/homebrew/lib
    # Construct the path with the specified LLVM version
    LLVM_VERSIONED_PREFIX := /opt/homebrew/opt/llvm@$(LLVM_VERSION)
    ifeq (,$(wildcard $(LLVM_VERSIONED_PREFIX)))
        # If not found, check for llvm at "/opt/homebrew/opt/llvm"
        LLVM_PREFIX := /opt/homebrew/opt/llvm
        ifeq (,$(wildcard $(LLVM_PREFIX)))
            # If still not found, use the latest version in "/opt/homebrew/Cellar/llvm/*"
            LLVM_PREFIX := $(shell ls -d /opt/homebrew/Cellar/llvm/* | tail -1)
        endif
    else
        # If the versioned llvm directory is found, use it
        LLVM_PREFIX := $(LLVM_VERSIONED_PREFIX)
    endif
endif

# Test if llvm-config version is correct
LLVM_CONFIG := $(LLVM_PREFIX)/bin/llvm-config
LLVM_VERSION_CMD := $(shell $(LLVM_CONFIG) --version)
LLVM_MAJOR_VERSION := $(shell echo ${LLVM_VERSION_CMD} | cut -d. -f1)

MLIR_SYS_$(LLVM_VERSION)0_PREFIX ?= $(LLVM_PREFIX)
LLVM_SYS_$(LLVM_VERSION)0_PREFIX ?= $(LLVM_PREFIX)
TABLEGEN_$(LLVM_VERSION)0_PREFIX ?= $(LLVM_PREFIX)

export MLIR_SYS_$(LLVM_VERSION)0_PREFIX
export LLVM_SYS_$(LLVM_VERSION)0_PREFIX
export TABLEGEN_$(LLVM_VERSION)0_PREFIX

HOST_TARGET := $(shell rustc -Vv | grep "host" | awk '{print $$2}')

.PHONY: check_llvm_version
check_llvm_version:
	@if [ $(LLVM_MAJOR_VERSION) -ne $(LLVM_VERSION) ]; then \
		echo "LLVM version mismatch: $(LLVM_MAJOR_VERSION) != $(LLVM_VERSION)"; \
		exit 1; \
	fi

.PHONY: setup
setup: check_llvm_version
	@echo "Installing dependencies..."
	@rustup default nightly-2024-05-02
	@rustup component add clippy
	@rustup component add rustfmt
	@cargo install cargo-llvm-cov cargo-nextest cargo-tarpaulin
	@rustup component add llvm-tools-preview

.PHONY: setup-llvm
setup-llvm: 
	@echo "Installing LLVM..."
	@sudo ./utils/setup.sh

.PHONY: setup-local 
setup-local: setup-llvm setup

.PHONY: format
format:
	@echo "Running formatter..."
	@cargo fmt --all

.PHONY: format-check
format-check:
	@echo "Running formatter..."
	@cargo fmt --all -- --check

.PHONY: lint
lint:
	@echo "Running linter..."
	@cargo clippy --all-targets --all-features -- #-D warnings

.PHONY: github-checks
github-checks: format-check lint

.PHONY: code-coverage
code-coverage:
	@echo "Running code coverage..."
	@cargo tarpaulin --all --all-features --workspace --timeout 120 --out xml

.PHONY: code-coverage-html
code-coverage-html:
	@echo "Running code coverage..."
	@cargo tarpaulin --all --all-features --workspace --timeout 120 --out html

.PHONY: build
build:
	@echo "Building mlir-sys (unoptimized)..."
	@cargo build --all --all-features

.PHONY: clean
clean:
	@echo "Cleaning..."
	@cargo clean

.PHONY: build-release
build-release: format-check lint
	@echo "Building mlir-sys (release)..."
	@cargo build --release --all --all-features

.PHONY: run
run:
	@cargo run --release

.PHONY: test
test:
	@echo "Running test..."
	@cargo test

.PHONY: test-memory
test-memory:
	@echo "Running memory sanitizer($(HOST_TARGET))..."
	@RUSTFLAGS="-Z sanitizer=memory" cargo test --target $(HOST_TARGET)

.PHONY: test-address
test-address:
	@echo "Running address sanitizer($(HOST_TARGET))..."
	@RUSTFLAGS="-Z sanitizer=address" cargo test --target $(HOST_TARGET)

.PHONY: test-valgrind
test-valgrind:
	@echo "Running valgrind memory tester..."
	@./utils/valgrind.sh

.PHONY: test-all
test-all: test test-memory test-address test-valgrind