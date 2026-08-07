#pragma once
// CuMetal CUB shim: DeviceMergeSort — device-level merge sort.

#include <cuda_runtime.h>
#include <algorithm>
#include <iterator>
#include <vector>

extern "C" void* cumetalRuntimeGetHostPointer(const void* ptr, size_t count);

namespace cub {

namespace detail {
// These sorts run on the CPU, so the caller's stream has to be drained first and raw device
// pointers have to be mapped back to host-addressable ones (under
// CUMETAL_USE_METAL_DEVICE_ADDRESSES they are Metal GPU addresses the CPU cannot dereference).
template <typename IteratorT>
IteratorT host_iterator(IteratorT iterator) {
    return iterator;
}

template <typename T>
T* host_iterator(T* pointer) {
    return static_cast<T*>(cumetalRuntimeGetHostPointer(pointer, 0));
}
}  // namespace detail

struct DeviceMergeSort {
    // Sort keys in-place
    template <typename KeyIteratorT, typename CompareOpT>
    static cudaError_t SortKeys(void* d_temp_storage, size_t& temp_storage_bytes,
                                KeyIteratorT d_keys, int num_items,
                                CompareOpT compare_op, cudaStream_t stream = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        cudaStreamSynchronize(stream);
        auto keys = detail::host_iterator(d_keys);
        std::stable_sort(keys, keys + num_items, compare_op);
        return cudaSuccess;
    }

    // Sort key-value pairs in-place
    template <typename KeyIteratorT, typename ValueIteratorT, typename CompareOpT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 KeyIteratorT d_keys, ValueIteratorT d_items,
                                 int num_items, CompareOpT compare_op, cudaStream_t stream = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        cudaStreamSynchronize(stream);
        auto keys = detail::host_iterator(d_keys);
        auto items = detail::host_iterator(d_items);

        // Build index permutation and sort by keys
        std::vector<int> idx(num_items);
        for (int i = 0; i < num_items; i++) idx[i] = i;

        using KeyT = typename std::iterator_traits<KeyIteratorT>::value_type;
        using ValT = typename std::iterator_traits<ValueIteratorT>::value_type;

        std::vector<KeyT> keys_copy(keys, keys + num_items);
        std::vector<ValT> vals_copy(items, items + num_items);

        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            return compare_op(keys_copy[a], keys_copy[b]);
        });
        for (int i = 0; i < num_items; i++) {
            keys[i] = keys_copy[idx[i]];
            items[i] = vals_copy[idx[i]];
        }
        return cudaSuccess;
    }

    // Sort keys copy
    template <typename KeyInputIteratorT, typename KeyIteratorT, typename CompareOpT>
    static cudaError_t SortKeysCopy(void* d_temp_storage, size_t& temp_storage_bytes,
                                    KeyInputIteratorT d_input_keys, KeyIteratorT d_output_keys,
                                    int num_items, CompareOpT compare_op, cudaStream_t stream = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        cudaStreamSynchronize(stream);
        auto input_keys = detail::host_iterator(d_input_keys);
        auto output_keys = detail::host_iterator(d_output_keys);
        std::copy(input_keys, input_keys + num_items, output_keys);
        std::stable_sort(output_keys, output_keys + num_items, compare_op);
        return cudaSuccess;
    }
};

} // namespace cub
