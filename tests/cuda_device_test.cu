#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void write_value(int *output) {
    *output = 42;
}

bool check(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
    return false;
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice || device_count == 0) {
        std::cerr << "SKIP: CUDA build is available, but no CUDA device was found\n";
        return 77;
    }
    if (!check(count_status, "cudaGetDeviceCount")) {
        return 1;
    }

    int *device_value = nullptr;
    if (!check(cudaMalloc(&device_value, sizeof(int)), "cudaMalloc")) {
        return 1;
    }

    write_value<<<1, 1>>>(device_value);
    bool success = check(cudaGetLastError(), "kernel launch");

    int host_value = 0;
    success = check(cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost),
                    "cudaMemcpy") &&
              success;
    success = check(cudaFree(device_value), "cudaFree") && success;

    if (!success || host_value != 42) {
        std::cerr << "CUDA device smoke test failed\n";
        return 1;
    }
    return 0;
}
