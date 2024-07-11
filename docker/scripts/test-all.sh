#!/usr/bin/env bash

OUTPUT_DIR="./.output"
mkdir -p $OUTPUT_DIR
echo "Running mlir-rs tests, results will be saved in $OUTPUT_DIR";
make test > $OUTPUT_DIR/test_result.txt
make test-address > $OUTPUT_DIR/test-address_result.txt
make test-memory > $OUTPUT_DIR/test-memory_result.txt
make test-valgrind > $OUTPUT_DIR/test-valgrind_result.txt
echo "Test results are in $OUTPUT_DIR"