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
> Sometimes the first 1 - 2 runs can result in errors, however subsequent runs will have no issues. This is intermittent, so you may not run into this.

```bash
cd docker && docker build -t mlir-rs-test:latest .
docker compose run mlir-rs-test
# After running docker compose run you'll be placed in the container
make test
make test-address
make test-memory
make test-valgrind
```

### MacOS - On Host
![If only you knew how bad things really are](https://static1.cbrimages.com/wordpress/wp-content/uploads/2023/01/this-is-fine.jpg)