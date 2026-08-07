#pragma once

// Texture-object support for the Metal backend.
//
// Metal has real texture objects, but CUDA texture objects carry addressing/filtering state that
// does not map onto MTLTexture one-for-one, and CUDA code reaches them through tex2D() rather than
// through a sampler argument. Instead of translating, a texture object here is the *device address*
// of a cumetalTextureRecord_t, and tex2D() samples the underlying pitched allocation directly.
// Real CUDA also makes a texture object an opaque device-resident handle, so this keeps the ABI
// shape callers expect: an unsigned long long passed by value into a kernel.
//
// Because the handle is dereferenced on the GPU, cudaMalloc has to hand out real MTLBuffer GPU
// addresses; run with CUMETAL_USE_METAL_DEVICE_ADDRESSES=1. With the default CPU shared mapping a
// kernel can hold the handle but not read through it.
//
// Only cudaResourceTypePitch2D is supported. cudaCreateTextureObject rejects the other resource
// types rather than returning a handle that would fault at first sample.

#include <math.h>

#include "cuda_runtime.h"

typedef struct cumetalTextureRecord {
    const void* data;
    unsigned int width;
    unsigned int height;
    unsigned int pitch_bytes;
    int filter_linear;
    int normalized_coords;
    int channel_bits;
    int channel_kind;
} cumetalTextureRecord_t;

// The record itself is shared: the host fills one in and copies it to the device. The samplers
// below only exist under device compilation, where __device__ is defined.
#if defined(__clang__) && defined(__CUDA__)

// Addressing is clamp-only. CUDA honours Wrap and Mirror for normalized coordinates only, so with
// normalizedCoords = 0 those modes already collapse to clamp on real hardware.
template <typename _ElementType>
static __device__ __forceinline__ _ElementType cumetal_texel_at(const cumetalTextureRecord_t* texture,
                                                                int x, int y) {
    const int max_x = static_cast<int>(texture->width) - 1;
    const int max_y = static_cast<int>(texture->height) - 1;
    x = x < 0 ? 0 : (x > max_x ? max_x : x);
    y = y < 0 ? 0 : (y > max_y ? max_y : y);
    const char* row = static_cast<const char*>(texture->data) +
                      static_cast<unsigned long long>(y) * texture->pitch_bytes;
    return reinterpret_cast<const _ElementType*>(row)[x];
}

// Mirrors CUDA's linear filtering, including the 8-bit fixed-point quantization of the
// interpolation weights described in the CUDA C Programming Guide's texture-fetching appendix.
// Interpolating at full float precision would make this *more* accurate than the hardware it
// stands in for, which shows up as unexplained divergence when comparing against a real CUDA run.
template <typename _ElementType>
static __device__ __forceinline__ _ElementType cumetal_sample_linear(const cumetalTextureRecord_t* texture,
                                                                     float x, float y) {
    const float shifted_x = x - 0.5f;
    const float shifted_y = y - 0.5f;
    const float floor_x = floorf(shifted_x);
    const float floor_y = floorf(shifted_y);
    const int base_x = static_cast<int>(floor_x);
    const int base_y = static_cast<int>(floor_y);
    const float alpha = floorf((shifted_x - floor_x) * 256.0f) * (1.0f / 256.0f);
    const float beta = floorf((shifted_y - floor_y) * 256.0f) * (1.0f / 256.0f);

    const float top_left = static_cast<float>(cumetal_texel_at<_ElementType>(texture, base_x, base_y));
    const float top_right = static_cast<float>(cumetal_texel_at<_ElementType>(texture, base_x + 1, base_y));
    const float bottom_left = static_cast<float>(cumetal_texel_at<_ElementType>(texture, base_x, base_y + 1));
    const float bottom_right =
        static_cast<float>(cumetal_texel_at<_ElementType>(texture, base_x + 1, base_y + 1));

    return static_cast<_ElementType>((1.0f - alpha) * (1.0f - beta) * top_left +
                                     alpha * (1.0f - beta) * top_right +
                                     (1.0f - alpha) * beta * bottom_left +
                                     alpha * beta * bottom_right);
}

template <typename _ElementType>
static __device__ __forceinline__ _ElementType tex2D(cudaTextureObject_t object, float x, float y) {
    const cumetalTextureRecord_t* texture = reinterpret_cast<const cumetalTextureRecord_t*>(object);
    if (texture->normalized_coords != 0) {
        x *= static_cast<float>(texture->width);
        y *= static_cast<float>(texture->height);
    }
    if (texture->filter_linear != 0) {
        return cumetal_sample_linear<_ElementType>(texture, x, y);
    }
    return cumetal_texel_at<_ElementType>(texture, static_cast<int>(floorf(x)),
                                          static_cast<int>(floorf(y)));
}

#endif  // device code section
