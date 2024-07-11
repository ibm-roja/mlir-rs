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

The exact steps being run can be found in the `docker/scripts/test-all.sh` script.

Resulting logs will be stored in the host repo directory under `.output`.

```bash
cd docker && docker compose run mlir-rs-test
```