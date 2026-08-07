// Texture objects are device addresses of a cumetalTextureRecord_t (see runtime/api/texture_types.h),
// so tex2D() dereferences the handle on the GPU. That only works when cudaMalloc hands out real
// MTLBuffer GPU addresses, which is why the runner sets CUMETAL_USE_METAL_DEVICE_ADDRESSES=1.

#include "cuda_runtime.h"

#include <cstdio>
#include <cmath>

__global__ void sample_point(cudaTextureObject_t texture, float* out, int width) {
    const int x = static_cast<int>(threadIdx.x);
    out[x] = tex2D<float>(texture, static_cast<float>(x) + 0.5f, 0.5f);
}

__global__ void sample_linear(cudaTextureObject_t texture, float* out) {
    // Midway between texel 0 and texel 1, so the result is their average.
    out[0] = tex2D<float>(texture, 1.0f, 0.5f);
}

__global__ void sample_clamped(cudaTextureObject_t texture, float* out, int width) {
    // Off both edges: CUDA clamps to the first and last texel.
    out[0] = tex2D<float>(texture, -4.5f, 0.5f);
    out[1] = tex2D<float>(texture, static_cast<float>(width) + 3.5f, 0.5f);
}

static cudaTextureObject_t make_texture(const float* device_row, int width, size_t pitch,
                                        cudaTextureFilterMode filter) {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceDesc::cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = const_cast<float*>(device_row);
    resource.res.pitch2D.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    resource.res.pitch2D.width = static_cast<size_t>(width);
    resource.res.pitch2D.height = 1;
    resource.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = filter;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;

    cudaTextureObject_t texture = 0;
    const cudaError_t err = cudaCreateTextureObject(&texture, &resource, &description, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaCreateTextureObject returned %d\n", err);
        return 0;
    }
    return texture;
}

int main() {
    const int width = 4;
    const float host_row[width] = {10.0f, 20.0f, 30.0f, 40.0f};

    float* device_row = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_row), sizeof(host_row));
    cudaMemcpy(device_row, host_row, sizeof(host_row), cudaMemcpyHostToDevice);

    float* device_out = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_out), sizeof(host_row));

    cudaTextureObject_t point = make_texture(device_row, width, sizeof(host_row), cudaFilterModePoint);
    if (point == 0) {
        return 1;
    }
    sample_point<<<1, width>>>(point, device_out, width);
    cudaDeviceSynchronize();

    float got[width] = {};
    cudaMemcpy(got, device_out, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < width; ++i) {
        if (got[i] != host_row[i]) {
            std::fprintf(stderr, "FAIL: point sample [%d] = %f, expected %f\n", i, got[i], host_row[i]);
            return 1;
        }
    }

    sample_clamped<<<1, 1>>>(point, device_out, width);
    cudaDeviceSynchronize();
    cudaMemcpy(got, device_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (got[0] != host_row[0] || got[1] != host_row[width - 1]) {
        std::fprintf(stderr, "FAIL: clamped samples = %f, %f; expected %f, %f\n",
                     got[0], got[1], host_row[0], host_row[width - 1]);
        return 1;
    }

    cudaTextureObject_t linear = make_texture(device_row, width, sizeof(host_row), cudaFilterModeLinear);
    if (linear == 0) {
        return 1;
    }
    sample_linear<<<1, 1>>>(linear, device_out);
    cudaDeviceSynchronize();
    cudaMemcpy(got, device_out, sizeof(float), cudaMemcpyDeviceToHost);
    // CUDA quantizes the interpolation weight to 8 bits, so an exact 0.5 weight stays exact.
    const float expected = 0.5f * (host_row[0] + host_row[1]);
    if (std::fabs(got[0] - expected) > 1e-4f) {
        std::fprintf(stderr, "FAIL: linear sample = %f, expected %f\n", got[0], expected);
        return 1;
    }

    cudaDestroyTextureObject(point);
    cudaDestroyTextureObject(linear);
    cudaFree(device_out);
    cudaFree(device_row);

    std::printf("PASS: texture objects sample on the GPU\n");
    return 0;
}
