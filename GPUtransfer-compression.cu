#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nvcomp.hpp>
#include <nvcomp/gdeflate.hpp>

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

    std::cout << "Loaded checkpoint with " << weights.size() << " weights." << std::endl;
}

int main() {
    const std::string checkpointFile = "/work/hdd/bdof/nkanamarla/models/sharded_model_download/dl_state_dict.bin";
    std::vector<float> weights;
    loadCheckpointFromDisk(checkpointFile, weights);

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "This program requires at least two GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    int srcDevice = 0;
    int dstDevice = 1;

    // Enable peer access between GPUs
    int canAccessPeer;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, srcDevice, dstDevice));
    if (canAccessPeer) {
        CHECK_CUDA(cudaSetDevice(srcDevice));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dstDevice, 0));
    } else {
        std::cerr << "Peer access not supported between GPU " << srcDevice << " and GPU " << dstDevice << "." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Setup peer access between source and destination GPU." << std::endl;

    // Set up CUDA stream
    CHECK_CUDA(cudaSetDevice(srcDevice));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Allocate and copy data to source GPU
    float* d_srcWeights;
    size_t dataSize = weights.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_srcWeights, dataSize));
    CHECK_CUDA(cudaMemcpy(d_srcWeights, weights.data(), dataSize, cudaMemcpyHostToDevice));
    std::cout << "Weights transferred to source GPU " << srcDevice << " of size " << dataSize << " bytes." << std::endl;

    {
    // Set up GDeflate manager
    nvcompBatchedGdeflateOpts_t gdeflate_opts;
    const size_t chunk_size = 1 << 16; // 64 KB chunks
    nvcomp::GdeflateManager gdeflate_manager(
        chunk_size,
        gdeflate_opts,
        stream,
        srcDevice,
        nvcomp::ChecksumPolicy::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE
    );

    // Compress data on source GPU
    nvcomp::CompressionConfig comp_config = gdeflate_manager.configure_compression(dataSize);
    uint8_t* d_compressedData;
    CHECK_CUDA(cudaMallocAsync(&d_compressedData, comp_config.max_compressed_buffer_size, stream));
    try {
        gdeflate_manager.compress(
            reinterpret_cast<uint8_t*>(d_srcWeights),
            d_compressedData,
            comp_config
    );
    } catch (const std::exception& e) {
        std::cerr << "Compression failed: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Compressed data on source GPU of size " << comp_config.max_compressed_buffer_size << " bytes." << std::endl;

    // Allocate memory on destination GPU
    CHECK_CUDA(cudaSetDevice(dstDevice));
    uint8_t* d_dstCompressedData;
    CHECK_CUDA(cudaMalloc(&d_dstCompressedData, comp_config.max_compressed_buffer_size));

    // Start timing
    CHECK_CUDA(cudaSetDevice(srcDevice));
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Transfer compressed data between GPUs
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dstCompressedData, dstDevice, d_compressedData, srcDevice, comp_config.max_compressed_buffer_size, stream));

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Compressed GPU-to-GPU transfer took " << milliseconds << " ms." << std::endl;
    CHECK_CUDA(cudaFree(d_compressedData));
    CHECK_CUDA(cudaSetDevice(dstDevice));
    CHECK_CUDA(cudaFree(d_dstCompressedData));
    } // gdeflate_manager is destroyed here

    // Cleanup
    CHECK_CUDA(cudaFree(d_srcWeights));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaSetDevice(srcDevice));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(dstDevice));

    std::cout << "Cleanup complete. Exiting program." << std::endl;
    return 0;
}