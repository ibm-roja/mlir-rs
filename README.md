# mlir-rs 

## Memory/Address Testing

### Windows/Linux - On Host
Memory testing on a x86/ARM Linux machine is easier as we do not require a docker container to run the tests. Run these steps to execute memory/address sanitizers and valgrind.
```bash
make setup-local
make test
make test-address
make test-memory
make test-valgrind
```

### MacOS - Container
Memory testing on M series macbooks is more involved, but still possible. Our solution is to use a docker container running ubuntu arm, and building said docker image before running any tests. We mount the mlir-rs repo into the container, allowing the container to access the make steps we want to run. 

Docker will perform the following actions
1. Run `cargo test` with address sanitizer enabled
2. Run `cargo test` with memory sanitizer enabled
3. Run `cargo test` using valgrind

The exact scripts being run can be found in the `docker/scripts/` directory.

> [!TIP]  
> Building the docker image will run the same setup as make setup-local, so no need to run that again.

> [!WARNING]  
> Sometimes the first 1 - 2 runs can result in linker errors, however subsequent runs will have no issues. This is intermittent, so you may not run into this.

```bash
docker compose run mlir-rs-test
```