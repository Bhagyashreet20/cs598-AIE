#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

// Transfer two checkpoint shards between two source GPUs to one destination GPU on the same node and measure the time (0.5 ms)
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
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

    std::cout << "Loaded checkpoint: " << filename << " with " << weights.size() << " weights." << std::endl;
}

int main() {
    // File paths for the two files to be transferred
    const std::string checkpointFile = "/work/hdd/bdof/nkanamarla/models/LLAMA3checkpointbinformatDS";

    // Load files into memory
    std::vector<float> weights;
    loadCheckpointFromDisk(checkpointFile1, weights);
    auto middle = weights.begin() + weights.size() / 2;
    std::vector<uint8_t> weights1(weights.begin(), middle);
    std::vector<uint8_t> weights2(middle, weights.end());

    // Set the GPUs to use
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 3) {
        std::cerr << "This program requires at least three GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    int srcDevice1 = 0; // First source GPU
    int srcDevice2 = 1; // Second source GPU
    int dstDevice = 2;  // Destination GPU

    size_t dataSize1 = weights1.size() * sizeof(float);
    size_t dataSize2 = weights2.size() * sizeof(float);

    // Allocate memory and copy data on source GPU 1
    CHECK_CUDA(cudaSetDevice(srcDevice1));
    float* d_srcWeights1;
    CHECK_CUDA(cudaMalloc(&d_srcWeights1, dataSize1));
    CHECK_CUDA(cudaMemcpy(d_srcWeights1, weights1.data(), dataSize1, cudaMemcpyHostToDevice));

    // Allocate memory and copy data on source GPU 2
    CHECK_CUDA(cudaSetDevice(srcDevice2));
    float* d_srcWeights2;
    CHECK_CUDA(cudaMalloc(&d_srcWeights2, dataSize2));
    CHECK_CUDA(cudaMemcpy(d_srcWeights2, weights2.data(), dataSize2, cudaMemcpyHostToDevice));
    std::cout << "Allocated memory and transferred data on source GPUs" << std::endl;

    // Enable peer access between GPUs
    for (int src : {srcDevice1, srcDevice2}) {
        int canAccessPeer;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, src, dstDevice)); // Check peer access from source to destination
        if (canAccessPeer) {
            CHECK_CUDA(cudaSetDevice(src));
            CHECK_CUDA(cudaDeviceEnablePeerAccess(dstDevice, 0)); // Enable peer access from source to destination
        } else {
            std::cerr << "Peer access not supported between GPU " << src << " and GPU " << dstDevice << "." << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Setup peer access on source GPUs and destination GPUs" << std::endl;

    // Allocate memory on the destination GPU
    CHECK_CUDA(cudaSetDevice(dstDevice));
    float* d_dstWeights1;
    float* d_dstWeights2;
    CHECK_CUDA(cudaMalloc(&d_dstWeights1, dataSize1));
    CHECK_CUDA(cudaMalloc(&d_dstWeights2, dataSize2));
    std::cout << "Allocated memory on destination GPUs" << std::endl;

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Create CUDA events for timing
    cudaEvent_t start1, stop1;
    cudaEvent_t start2, stop2;
    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));
    std::cout << "Created CUDA streams and events" << std::endl;

    // Start timing
    CHECK_CUDA(cudaEventRecord(start1, stream1));
    CHECK_CUDA(cudaEventRecord(start2, stream2));

    // Transfer data using streams
    CHECK_CUDA(cudaSetDevice(srcDevice1));
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dstWeights1, dstDevice, d_srcWeights1, srcDevice1, dataSize1, stream1));
    CHECK_CUDA(cudaSetDevice(srcDevice2));
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dstWeights2, dstDevice, d_srcWeights2, srcDevice2, dataSize2, stream2));

    // Synchronize streams and record stop events on respective streams
    CHECK_CUDA(cudaEventRecord(stop1, stream1));
    CHECK_CUDA(cudaEventRecord(stop2, stream2));

    // Stop timing
    CHECK_CUDA(cudaEventSynchronize(stop1));
    CHECK_CUDA(cudaEventSynchronize(stop2));

    float milliseconds1 = 0, milliseconds2 = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds1, start1, stop1));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds2, start2, stop2));
    // Take the maximum time as the elapsed transfer time
    float maxElapsedTime = std::max(milliseconds1, milliseconds2);
    std::cout << "Parallel GPU-to-GPU transfer (with CUDA streams) took " << maxElapsedTime << " ms." << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_srcWeights1));
    CHECK_CUDA(cudaFree(d_srcWeights2));
    CHECK_CUDA(cudaFree(d_dstWeights1));
    CHECK_CUDA(cudaFree(d_dstWeights2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaEventDestroy(stop1));
    CHECK_CUDA(cudaEventDestroy(stop2));

    // Disable peer access
    for (int src : {srcDevice1, srcDevice2}) {
        CHECK_CUDA(cudaSetDevice(src));
        CHECK_CUDA(cudaDeviceDisablePeerAccess(dstDevice));
    }

    std::cout << "Cleanup complete. Exiting program." << std::endl;
    return 0;
}
