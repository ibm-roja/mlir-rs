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

> [!TIP]  
> Building the docker image will run the same setup as make setup-local, so no need to run that again.

> [!WARNING]  
> Sometimes the first 1 - 2 runs can result in linker errors, however subsequent runs will have no issues. This is intermittent, so you may not run into this.

```bash
# If image is not yet built. build it.
cd docker && docker build -t mlir-rs-test:latest .
# Otherwise run docker compose up from the docker directory.
docker compose up
```