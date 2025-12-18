#pragma once

#include "inferencer/inferencer_api.hpp"

#include <ament_index_cpp/get_package_prefix.hpp>
#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace nnc {

class DynamicInferencer final : public Inferencer {
public:
  explicit DynamicInferencer(const std::string& backend)
  {
    // lib<backend>_inferencer.so, installed into the inferencer package lib/
    const std::string lib_name = "lib" + backend + "_inferencer.so";
    const auto prefix = ament_index_cpp::get_package_prefix("inferencer");
    const std::filesystem::path lib_path = std::filesystem::path(prefix) / "lib" / lib_name;

    dll_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dll_) {
      throw std::runtime_error("dlopen failed for " + lib_path.string() + ": " + dlerror());
    }

    // Load C symbols (from inferencer_c.h ABI)
    create_    = load_sym<CreateFn>("createInferencer");
    destroy_   = load_sym<DestroyFn>("deleteInferencer");
    load_model_= load_sym<LoadModelFn>("loadModel");
    get_in_    = load_sym<GetBufFn>("getInputBuffer");
    get_out_   = load_sym<GetBufFn>("getOutputBuffer");
    infer_     = load_sym<InferFn>("infer");
    err_       = load_sym<ErrFn>("getErrorString");

    handle_ = create_(nullptr);
    if (!handle_) {
      throw std::runtime_error("createInferencer returned null for " + lib_name);
    }
  }

  DynamicInferencer(const DynamicInferencer&) = delete;
  DynamicInferencer& operator=(const DynamicInferencer&) = delete;

  ~DynamicInferencer() override
  {
    if (handle_ && destroy_) destroy_(handle_);
    if (dll_) dlclose(dll_);
  }

  bool loadModel(const std::string& path) override { return load_model_(handle_, path.c_str()); }

  unsigned getInputBuffer(const std::string& name, void** buffer) override
  {
    return get_in_(handle_, name.c_str(), buffer);
  }

  unsigned getOutputBuffer(const std::string& name, void** buffer) override
  {
    return get_out_(handle_, name.c_str(), buffer);
  }

  bool infer() override { return infer_(handle_); }

  std::string getErrorString() const override
  {
    const char* s = err_(handle_);
    return s ? std::string(s) : std::string();
  }

private:
  template <typename T>
  T load_sym(const char* name)
  {
    dlerror();  // clear
    void* p = dlsym(dll_, name);
    const char* e = dlerror();
    if (e || !p) {
      throw std::runtime_error(std::string("dlsym failed for ") + name + ": " + (e ? e : ""));
    }
    return reinterpret_cast<T>(p);
  }

  using CreateFn    = void* (*)(void*);
  using DestroyFn   = void (*)(void*);
  using LoadModelFn = bool (*)(void*, const char*);
  using GetBufFn    = unsigned (*)(void*, const char*, void**);
  using InferFn     = bool (*)(void*);
  using ErrFn       = const char* (*)(void*);

  void* dll_{nullptr};
  void* handle_{nullptr};

  CreateFn create_{nullptr};
  DestroyFn destroy_{nullptr};
  LoadModelFn load_model_{nullptr};
  GetBufFn get_in_{nullptr};
  GetBufFn get_out_{nullptr};
  InferFn infer_{nullptr};
  ErrFn err_{nullptr};
};

} // namespace nnc
