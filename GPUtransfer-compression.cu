#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nvcomp.hpp>
#include <nvcomp/cascaded.hpp>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void memoryCheck(int GPUnum) {
    size_t free_memory, total_memory;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    
    if (error != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(error) << std::endl;
    }
    
    std::cout << "For GPU " << GPUnum << " Free GPU memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "For GPU " << GPUnum << " Total GPU memory: " << total_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "For GPU " << GPUnum << " Used GPU memory: " << (total_memory - free_memory) / (1024 * 1024) << " MB" << std::endl;
}

void loadCheckpointFromDisk(const std::string& filename, std::vector<uint8_t>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    data.resize(fileSize);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);

    if (!file) {
        std::cerr << "Failed to read checkpoint file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Loaded checkpoint of size " << data.size() << " bytes in CPU memory." << std::endl;
}

int main() {
    // Load checkpoint from disk and split for test 
    const std::string checkpointFile = "/work/hdd/bdof/nkanamarla/models/LLAMA3checkpointbinformat/LLAMA3checkpoint.bin";
    std::vector<uint8_t> weights;
    loadCheckpointFromDisk(checkpointFile, weights);

    // Split up checkpoint code for next part
    auto middle = weights.begin() + weights.size() / 2;
    std::vector<uint8_t> weightsFirstHalf(weights.begin(), middle);
    std::vector<uint8_t> weightsSecondHalf(middle, weights.end());

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
    memoryCheck(srcDevice);

    // Set up CUDA stream
    CHECK_CUDA(cudaSetDevice(srcDevice));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Allocate and copy data to source GPU
    uint8_t* d_srcWeights;
    size_t dataSize = weightsFirstHalf.size();
    CHECK_CUDA(cudaMalloc(&d_srcWeights, dataSize));
    CHECK_CUDA(cudaMemcpy(d_srcWeights, weightsFirstHalf.data(), dataSize, cudaMemcpyHostToDevice));
    std::cout << "Shard weights transferred to source GPU " << srcDevice << " of size " << dataSize << " bytes." << std::endl;
    memoryCheck(srcDevice);

    {
    // Start timing
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Set up compression manager
    nvcompBatchedCascadedOpts_t cascade_options;
    cascade_options.type =  nvcomp::TypeOf<uint8_t>();
    cascade_options.num_RLEs = 1;
    cascade_options.num_deltas = 1;
    cascade_options.use_bp = 1;
    const size_t chunk_size = 1 << 22; // 4 MB chunks
    nvcomp::CascadedManager cascade_manager(
        chunk_size,
        cascade_options,
        stream
    );

    // Compress data on source GPU
    nvcomp::CompressionConfig comp_config = cascade_manager.configure_compression(dataSize);
    uint8_t* d_compressedData;
    comp_config.max_compressed_buffer_size = dataSize * 0.1; // nvCOMP is too conservative with the predicted compression size 
    std::cout << "Compression max possible size in bytes " << comp_config.max_compressed_buffer_size << std::endl;
    CHECK_CUDA(cudaMallocAsync(&d_compressedData, comp_config.max_compressed_buffer_size, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    try {
        cascade_manager.compress(
            d_srcWeights,
            d_compressedData,
            comp_config
    );
    } catch (const std::exception& e) {
        std::cerr << "Compression failed: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    size_t compressed_size = cascade_manager.get_compressed_output_size(d_compressedData);
    memoryCheck(srcDevice);

    // Allocate memory on destination GPU
    CHECK_CUDA(cudaSetDevice(dstDevice));
    uint8_t* d_dstCompressedData;
    CHECK_CUDA(cudaMalloc(&d_dstCompressedData, compressed_size));
    memoryCheck(dstDevice);

    // Transfer compressed data between GPUs
    CHECK_CUDA(cudaSetDevice(srcDevice));
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dstCompressedData, dstDevice, d_compressedData, srcDevice, compressed_size, stream));

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Compressed data on GPU of size " << compressed_size << " bytes for a compression ratio of " << dataSize/compressed_size << " X." << std::endl;
    std::cout << "Data Compression and GPU-to-GPU data transfer took " << milliseconds << " ms." << std::endl;
    CHECK_CUDA(cudaFree(d_compressedData));
    CHECK_CUDA(cudaSetDevice(dstDevice));
    CHECK_CUDA(cudaFree(d_dstCompressedData));
    } // cascade_manager is destroyed here

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