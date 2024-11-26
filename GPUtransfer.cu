#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void loadCheckpointFromDisk(const std::string& filename, std::vector<float>& weights) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    weights.resize(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(weights.data()), fileSize);

    if (!file) {
        std::cerr << "Failed to read checkpoint file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Loaded checkpoint with " << weights.size() << " weights." << std::endl;
}

int main() {
    // Model checkpoint filename
    const std::string checkpointFile = "/work/hdd/bdof/nkanamarla/models/LLAMA3download/model_checkpoint.bin";

    // Load model weights from disk
    std::vector<float> weights;
    loadCheckpointFromDisk(checkpointFile, weights);

    // Set the GPUs to use
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "This program requires at least two GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    int srcDevice = 0;
    int dstDevice = 1;

    // Allocate memory on source GPU
    CHECK_CUDA(cudaSetDevice(srcDevice));
    float* d_srcWeights;
    size_t dataSize = weights.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_srcWeights, dataSize));
    CHECK_CUDA(cudaMemcpy(d_srcWeights, weights.data(), dataSize, cudaMemcpyHostToDevice));

    std::cout << "Weights transferred to GPU " << srcDevice << "." << std::endl;

    // Enable peer access between GPUs
    int canAccessPeer;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, dstDevice, srcDevice));
    if (canAccessPeer) {
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dstDevice, 0));
    } else {
        std::cerr << "Peer access not supported between GPU " << srcDevice << " and GPU " << dstDevice << "." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory on destination GPU
    CHECK_CUDA(cudaSetDevice(dstDevice));
    float* d_dstWeights;
    CHECK_CUDA(cudaMalloc(&d_dstWeights, dataSize));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Transfer data between GPUs
    CHECK_CUDA(cudaMemcpyPeer(d_dstWeights, dstDevice, d_srcWeights, srcDevice, dataSize));

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU-to-GPU transfer took " << milliseconds << " ms." << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_srcWeights));
    CHECK_CUDA(cudaFree(d_dstWeights));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaSetDevice(srcDevice));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(dstDevice));

    std::cout << "Cleanup complete. Exiting program." << std::endl;
    return 0;
}
