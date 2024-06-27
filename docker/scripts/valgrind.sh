#!/usr/bin/env bash

output=$(cargo test 2>&1)
echo "output: $output"
test_binary_path=$(echo "$output" | grep 'Running unittests' | awk '{print $4}')
echo "test_binary_path: $test_binary_path"
test_binary_path="${$test_binary_path//[()]/}"
echo "test_binary_path: $test_binary_path"
echo "Running valgrind on cargo test binary: $test_binary_path"
valgrind --leak-check=full --error-exitcode=1 --track-origins=yes --log-file="./valgrind.log" --show-leak-kinds=all "$test_binary_path"
cat ./valgrind.log