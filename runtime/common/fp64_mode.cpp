#include "fp64_mode.h"

#include "cumetal_diag.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace cumetal {

ptx::Fp64Mode current_fp64_mode() {
    static const ptx::Fp64Mode mode = [] {
        const char* value = std::getenv("CUMETAL_FP64_MODE");
        if (value == nullptr || value[0] == '\0') {
            return ptx::Fp64Mode::kEmulate;
        }
        std::string normalized(value);
        std::transform(
            normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (normalized == "emulate") {
            return ptx::Fp64Mode::kEmulate;
        }
        if (normalized == "native") {
            return ptx::Fp64Mode::kNative;
        }
        if (normalized == "warn") {
            return ptx::Fp64Mode::kWarn;
        }
        warn_once("invalid-fp64-mode",
                  "invalid CUMETAL_FP64_MODE='" + std::string(value) +
                      "'; expected 'emulate', 'native' or 'warn', using default 'emulate'");
        return ptx::Fp64Mode::kEmulate;
    }();
    return mode;
}

const char* fp64_mode_name(ptx::Fp64Mode mode) {
    switch (mode) {
        case ptx::Fp64Mode::kEmulate:
            return "emulate";
        case ptx::Fp64Mode::kNative:
            return "native";
        case ptx::Fp64Mode::kWarn:
            return "warn";
    }
    return "emulate";
}

}  // namespace cumetal
