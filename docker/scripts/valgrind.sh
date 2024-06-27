#!/usr/bin/env bash

output=$(cargo test 2>&1)
echo "output: $output" &&
raw_test_binary_path=$(echo "$output" | grep 'Running unittests src/lib.rs' )
echo "raw_test_binary_path: $raw_test_binary_path"
test_binary_path=$(echo "$raw_test_binary_path" | awk '{print $4}')
echo "test_binary_path: $test_binary_path"
test_binary_path="${test_binary_path//[()]/}"
echo "test_binary_path: $test_binary_path"
echo "Running valgrind on cargo test binary: $test_binary_path"
valgrind --leak-check=full --error-exitcode=1 --track-origins=yes --log-file="./valgrind.log" --show-leak-kinds=all "$test_binary_path"
cat ./valgrind.log