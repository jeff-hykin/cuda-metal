#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::air_emitter {

enum class EmitMode {
    kXcrun,
    kExperimentalContainer,
};

struct EmitOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    EmitMode mode = EmitMode::kXcrun;
    bool overwrite = false;
    bool fallback_to_experimental = false;
    bool validate_output = true;
    bool run_xcrun_validate = false;
    std::string kernel_name = "vector_add";
    // Oldest macOS the metallib will load on, passed to `metal -mmacosx-version-min`. It sets
    // the container's PlatformMajor; Metal refuses a metallib newer than the running OS, so
    // leaving it at the SDK default pins every output to the build machine's macOS. Must match
    // the target the input IR was lowered for (LowerToLlvmOptions), or air-lld rejects the
    // air.version mismatch. Empty means the SDK default.
    std::string macos_deployment_target = "13.0";
};

struct EmitResult {
    bool ok = false;
    EmitMode mode_used = EmitMode::kXcrun;
    std::filesystem::path output;
    std::vector<std::string> logs;
    std::string error;
};

EmitResult emit_metallib(const EmitOptions& options);

}  // namespace cumetal::air_emitter
