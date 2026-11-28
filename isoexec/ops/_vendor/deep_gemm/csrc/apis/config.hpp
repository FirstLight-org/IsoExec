#pragma once

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../runtime/runtime.hpp"

namespace deep_gemm::config {

static void register_apis(pybind11::module_& m) {
    // IsoExec: the cache-key signature is mandatory, so this copy can never share cubins with upstream.
    m.def("init", [&](const std::string& library_root_path, const std::string& isoexec_signature) {
        init_jit(library_root_path, isoexec_signature);
    }, pybind11::arg("library_root_path"), pybind11::arg("isoexec_signature"));
    // IsoExec: report the identity the JIT compiles under (include root, cache namespace, cache
    // directories) without constructing the CUDA runtime, so callers can fail closed on CPU.
    m.def("isoexec_jit_config", []() {
        DJ_HOST_ASSERT(jit_config.has_value(), "IsoExec DeepGEMM JIT is not initialized");
        pybind11::dict out;
        pybind11::list include_dirs;
        for (const auto& dir : jit_config->include_dirs)
            include_dirs.append(dir.string());
        pybind11::list cache_dirs;
        for (const auto& dir : deep_jit::DiskCache::from_env(deep_jit::Env(jit_config->env_prefix)).paths)
            cache_dirs.append(dir.string());
        out["python_library_root"] = jit_config->python_library_root.string();
        out["include_dirs"] = include_dirs;
        pybind11::list include_prefixes;
        for (const auto& prefix : jit_config->include_prefixes)
            include_prefixes.append(prefix);
        out["include_prefixes"] = include_prefixes;
        out["env_prefix"] = jit_config->env_prefix;
        out["extra_signature"] = jit_config->extra_signature;
        out["cache_dirs"] = cache_dirs;
        return out;
    });
    m.def("shutdown", []() {
        runtime = deep_jit::LazyInit<Runtime>(nullptr);
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return runtime->get_tc_util();
    });
    m.def("set_pdl", [](const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
    });
    m.def("get_pdl", []() {
        return *jit->default_launch_options.enable_pdl;
    });
    m.def("set_early_launch", [&](const int& mode) {
        runtime->set_early_launch(mode);
    });
    m.def("get_early_launch", [&]() {
        return runtime->early_launch;
    });
    m.def("use_deterministic_algorithms", [&](const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_leaf_register_partials", [&](const bool& enabled) {
        heuristics_runtime->set_leaf_register_partials(enabled);
    });
    m.def("get_leaf_register_partials", [&]() {
        return heuristics_runtime->get_leaf_register_partials();
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
}

} // namespace deep_gemm::config
