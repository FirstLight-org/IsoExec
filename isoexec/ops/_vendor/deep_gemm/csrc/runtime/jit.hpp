#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

// IsoExec isolation: the JIT runtime of this vendored copy must never be shared with another
// DeepGEMM in the process (e.g. vLLM's bundled upstream). The extension is built with hidden
// visibility and -fno-gnu-unique (see _extension.py), so these inline globals are private to this
// shared object. Its environment prefix and cache-key signature are distinct from upstream's
// ("DG", "cutlass-<version>"), so neither process-wide DG_* settings nor cached cubins cross over.
inline constexpr const char* kIsoexecJitEnvPrefix = "ISOEXEC_DG";

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);
inline std::optional<deep_jit::Config> jit_config;

inline void init_jit(const std::string& library_root_path, const std::string& isoexec_signature) {
    DJ_HOST_ASSERT(not isoexec_signature.empty(), "IsoExec DeepGEMM JIT signature must not be empty");
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        kIsoexecJitEnvPrefix,
        isoexec_signature + "+cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir},
        {"deep_gemm/"});

    // Fail closed on re-initialization with a different identity: a live runtime must keep the
    // include root and cache namespace it was created with.
    if (jit_config.has_value()) {
        DJ_HOST_ASSERT(jit_config->python_library_root == config.python_library_root and
                       jit_config->extra_signature == config.extra_signature,
                       "IsoExec DeepGEMM JIT already initialized with root {} / signature {}; refusing {} / {}",
                       jit_config->python_library_root.string(), jit_config->extra_signature,
                       config.python_library_root.string(), config.extra_signature);
        return;
    }
    jit_config = config;

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        return runtime;
    });
}

}  // namespace deep_gemm
