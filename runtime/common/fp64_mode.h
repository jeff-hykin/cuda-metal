#pragma once

#include "cumetal/ptx/lower_to_llvm.h"

namespace cumetal {

// Process-wide policy selected by CUMETAL_FP64_MODE. The first query freezes the normalized
// mode so cache identity and the lowering that produced the entry cannot disagree if the
// environment is mutated later.
//
// Every path that lowers PTX must ask here rather than filling in LowerToLlvmOptions itself.
// The struct defaults to kNative, so a path that forgets emits real `double` ALU ops; the
// xcrun metal compiler accepts those, and the GPU driver then crashes its own compiler service
// at pipeline creation (XPC_ERROR_CONNECTION_INTERRUPTED) instead of reporting an error.
ptx::Fp64Mode current_fp64_mode();
const char* fp64_mode_name(ptx::Fp64Mode mode);

}  // namespace cumetal
