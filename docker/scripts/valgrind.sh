#!/usr/bin/env bash

export CARGO_TARGET_DIR="./valgrind-target"
output=$(cargo test --no-run -- --enable-debug)
executable_path=$(find $CARGO_TARGET_DIR/debug/deps -type f -executable)
echo "Running valgrind on cargo test binary: $executable_path"
valgrind --leak-check=full --error-exitcode=1 --track-origins=yes --log-file="./valgrind.log" --show-leak-kinds=all "$executable_path"
cat ./valgrind.log