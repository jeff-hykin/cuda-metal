#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::metal_backend {
class Buffer;
}

namespace cumetal::registration {

struct RegisteredKernel {
    std::string metallib_path;
    std::string kernel_name;
    std::vector<cumetalKernelArgInfo_t> arg_info;
    // Device printf format table (spec §5.3): non-empty iff kernel uses printf.
    // printf_formats[i] is the format string for format id i.
    std::vector<std::string> printf_formats;
    // Total bytes of static __shared__ memory (non-extern .shared declarations).
    // Used to call setThreadgroupMemoryLength when no dynamic shared memory is specified.
    std::size_t static_shared_bytes = 0;
    // Runtime-written __constant__ symbols this kernel reads, in the order of the hidden
    // constant-buffer arguments the PTX lowering appends after the explicit kernel arguments.
    std::vector<std::string> const_symbol_buffers;
    std::string provenance;
    std::string semantic_quality;
};

struct LaunchConfiguration {
    dim3 grid_dim{};
    dim3 block_dim{};
    std::size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
};

// Build the CUDA launch-argument ABI index from PTX entry signatures. This is
// intentionally a lightweight registration-time scan; full PTX parsing remains
// part of kernel lowering.
std::unordered_map<std::string, std::vector<cumetalKernelArgInfo_t>>
build_arg_info_index_from_ptx(const std::string& ptx_source);

// Resolve one launch ABI without allocating metadata for every other entry in
// a large fatbinary module. Returns false when the entry is absent or malformed.
bool find_arg_info_for_ptx_entry(const std::string& ptx_source,
                                 std::string_view entry_name,
                                 std::vector<cumetalKernelArgInfo_t>* out);

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out);
bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size);

// Device storage allocated for a __constant__ symbol at registration time, keyed by the
// mangled device name the PTX uses. The launch path binds this as a hidden kernel argument.
bool lookup_symbol_storage(const std::string& device_name,
                           std::shared_ptr<cumetal::metal_backend::Buffer>* out);
void clear();

struct PrewarmResult {
    std::size_t total = 0;
    std::size_t lowered = 0;
    std::vector<std::string> failed;
};

// Lower every registered kernel into the persistent JIT cache, not just the ones a given run
// happens to launch. Packaging uses this so a shipped cache covers every code path (IMU,
// depth, RGB inputs) instead of whichever branches the sample data took.
PrewarmResult prewarm_all_registered_kernels();

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin);
void** __cudaRegisterFatBinary2(const void* fat_cubin, ...);
void** __cudaRegisterFatBinary3(const void* fat_cubin, ...);
void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global);
void __cudaRegisterManagedVar(void** fat_cubin_handle,
                              void** host_var_ptr_address,
                              char* device_address,
                              const char* device_name,
                              int ext,
                              std::size_t size,
                              int constant,
                              int global);
cudaError_t __cudaPushCallConfiguration(dim3 grid_dim,
                                        dim3 block_dim,
                                        std::size_t shared_mem,
                                        cudaStream_t stream);
cudaError_t __cudaPopCallConfiguration(dim3* grid_dim,
                                       dim3* block_dim,
                                       std::size_t* shared_mem,
                                       void** stream);

}  // extern "C"
