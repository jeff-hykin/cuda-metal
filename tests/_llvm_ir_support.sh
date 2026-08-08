#!/usr/bin/env bash
# Shared guard for the cumetalc cases that read a .cu. Source, do not execute.
#
# cumetalc's CUDA front end hands clang's LLVM IR to LLVM's IRReader, and LLVM is an optional
# dependency: the root CMakeLists only warns when it is absent, and the distributed runtime is
# built with -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=ON on purpose, since nothing on the PTX-to-metallib
# path needs it and linking it drags in Homebrew dylibs. In that configuration these cases cannot
# run at all, which ctest should report as a skip rather than a failure.

run_cumetalc_or_skip() {
    local output status
    set +e
    output="$("$@" 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$output"
    if [ "$status" -ne 0 ]; then
        case "$output" in
            *"without LLVM"*)
                echo "SKIP: CuMetal built without LLVM, so the .cu front end is unavailable"
                exit 77
                ;;
        esac
        exit "$status"
    fi
}
