// Lowers every kernel a CUDA-linked library registers into the persistent JIT cache.
//
// Normal execution lowers a kernel the first time it is launched, so the cache a run leaves
// behind only covers the branches that run took. Redistributing that cache to machines with
// no Metal compiler needs every kernel, including the ones the sample data never reaches.
// Loading the library runs its __cudaRegisterFunction calls, which is enough to enumerate
// and lower all of them.

#include "registration.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: cumetal_prewarm <library.dylib> [more.dylib ...]\n"
                     "\n"
                     "Set CUMETAL_CACHE_DIR to control where the metallibs land.\n");
        return 2;
    }

    for (int i = 1; i < argc; ++i) {
        if (dlopen(argv[i], RTLD_NOW | RTLD_GLOBAL) == nullptr) {
            const char* reason = dlerror();
            std::fprintf(stderr, "cumetal_prewarm: failed to load %s: %s\n", argv[i],
                         reason != nullptr ? reason : "unknown error");
            return 1;
        }
    }

    const cumetal::registration::PrewarmResult result =
        cumetal::registration::prewarm_all_registered_kernels();

    std::printf("prewarmed %zu/%zu kernels\n", result.lowered, result.total);
    for (const std::string& kernel_name : result.failed) {
        std::printf("  failed: %s\n", kernel_name.c_str());
    }

    if (result.total == 0) {
        std::fprintf(stderr, "cumetal_prewarm: no kernels registered\n");
        return 1;
    }
    return result.failed.empty() ? 0 : 1;
}
