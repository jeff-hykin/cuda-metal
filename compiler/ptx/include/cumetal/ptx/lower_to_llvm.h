#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

// FP64 compilation mode (see spec §8.1 and --fp64 CLI flag).
enum class Fp64Mode {
    kNative,   // emit AIR FP64 instructions as-is (default)
    kEmulate,  // Dekker FP32-pair decomposition (~44-bit mantissa); default in runtime
    kWarn,     // same as kNative but emit a per-instruction warning for .f64 ops
};

// The AIR and Metal Shading Language versions a given macOS release's driver accepts.
// Apple renumbered 15 -> 26, so there is no 16..25.
struct MetalTargetVersions {
    int air_minor;
    int language_major;
    int language_minor;
    std::string triple;  // e.g. air64_v25-apple-macosx13.0.0
};

MetalTargetVersions metal_target_versions(std::string_view macos_deployment_target);

struct LowerToLlvmOptions {
    bool strict = false;
    std::string entry_name;
    std::string module_id = "cumetal.ptx.module";
    // Oldest macOS the emitted AIR will load on. Metal rejects a metallib whose AIR version
    // is newer than the running OS, so emitting the newest (the SDK default) would pin every
    // compiled kernel — and any cache shipped with a package — to the build machine's macOS.
    std::string macos_deployment_target = "13.0";
    // Overrides the triple derived from the deployment target. Empty means derive it.
    std::string target_triple;
    Fp64Mode fp64_mode = Fp64Mode::kNative;
};

struct LowerToLlvmResult {
    bool ok = false;
    std::string entry_name;
    std::string llvm_ir;
    std::vector<std::string> warnings;
    std::string error;
};

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx,
                                       const LowerToLlvmOptions& options = {});

// Return the runtime-written __constant__ symbols the named entry reads, in the order of the
// hidden constant-buffer arguments the lowering appends. Derived from the PTX alone so the launch
// path can bind them even when a cached metallib made lowering unnecessary.
std::vector<std::string> runtime_const_symbols_for_entry(std::string_view ptx,
                                                         std::string_view entry_name);

// Return the total bytes of static __shared__ memory required by the PTX.
// This is needed to call setThreadgroupMemoryLength at kernel launch time.
std::size_t compute_static_shared_bytes(std::string_view ptx,
                                        std::string_view entry_name = {});

}  // namespace cumetal::ptx
