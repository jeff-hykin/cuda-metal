// PTX indirect branch (brx.idx + .branchtargets) coverage.
//
// A switch whose arms select different pointers is what makes clang emit a jump table rather
// than folding the arms into arithmetic or a compare chain. The .branchtargets list is in case
// order but the blocks are emitted out of order, so an implementation that assumed physical
// order reads the wrong array and gets caught here rather than crashing.
//
// cuvslam::cuda::sba_imu::build_full_system_stage_3_kernel is the real-world instance: it was
// the only cuVSLAM kernel containing a brx, and because PTX modules are parsed as a unit, it
// took all 19 sba_imu kernels down with it.

#include <cstdio>

__global__ void switch_jump_table_kernel(const int* source0,
                                         const int* source1,
                                         const int* source2,
                                         const int* source3,
                                         const int* source4,
                                         const int* source5,
                                         const int* source6,
                                         const int* source7,
                                         const int* selector,
                                         int* out,
                                         int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int* chosen;
    switch (selector[i]) {
        case 0: chosen = source0; break;
        case 1: chosen = source1; break;
        case 2: chosen = source2; break;
        case 3: chosen = source3; break;
        case 4: chosen = source4; break;
        case 5: chosen = source5; break;
        case 6: chosen = source6; break;
        case 7: chosen = source7; break;
        default: chosen = source0; break;
    }
    out[i] = chosen[i] + selector[i];
}

int main() {
    constexpr int kCount = 64;
    constexpr int kSources = 8;

    int host_selector[kCount];
    int host_out[kCount];
    int host_sources[kSources][kCount];
    for (int i = 0; i < kCount; ++i) {
        // Modulo 9 so the default arm is exercised too.
        host_selector[i] = i % (kSources + 1);
        for (int s = 0; s < kSources; ++s) {
            host_sources[s][i] = s * 1000 + i;
        }
    }

    const int* device_sources[kSources];
    for (int s = 0; s < kSources; ++s) {
        int* buffer = nullptr;
        cudaMalloc(&buffer, sizeof(host_sources[s]));
        cudaMemcpy(buffer, host_sources[s], sizeof(host_sources[s]), cudaMemcpyHostToDevice);
        device_sources[s] = buffer;
    }

    int* device_selector = nullptr;
    int* device_out = nullptr;
    cudaMalloc(&device_selector, sizeof(host_selector));
    cudaMalloc(&device_out, sizeof(host_out));
    cudaMemcpy(device_selector, host_selector, sizeof(host_selector), cudaMemcpyHostToDevice);

    switch_jump_table_kernel<<<(kCount + 31) / 32, 32>>>(
        device_sources[0], device_sources[1], device_sources[2], device_sources[3],
        device_sources[4], device_sources[5], device_sources[6], device_sources[7],
        device_selector, device_out, kCount);
    const cudaError_t launch_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess) {
        std::printf("FAIL: switch_jump_table_kernel launch failed: %s\n",
                    cudaGetErrorString(launch_status));
        return 1;
    }
    cudaMemcpy(host_out, device_out, sizeof(host_out), cudaMemcpyDeviceToHost);

    int failures = 0;
    for (int i = 0; i < kCount; ++i) {
        const int selector = host_selector[i];
        const int source = selector < kSources ? selector : 0;
        const int expected = host_sources[source][i] + selector;
        if (host_out[i] != expected) {
            if (failures < 10) {
                std::printf("  i=%d selector=%d expected=%d got=%d\n", i, selector, expected,
                            host_out[i]);
            }
            ++failures;
        }
    }
    if (failures != 0) {
        std::printf("FAIL: %d of %d indirect-branch results are wrong\n", failures, kCount);
        return 1;
    }
    std::printf("PASS: PTX indirect branch selects the right target for all %d elements\n", kCount);
    return 0;
}
