#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cerr << "SKIP: no CUDA device was found\n";
        return 77;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
