#pragma once

#include <ament_index_cpp/get_package_prefix.hpp>
#include <dlfcn.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

namespace inferencer
{
class Inferencer
{
public:
  explicit Inferencer(const std::string & backend)
  {
    const std::string lib_name = "lib" + backend + "_inferencer.so";
    const std::string prefix = ament_index_cpp::get_package_prefix("inferencer");
    const std::filesystem::path lib_path = std::filesystem::path(prefix) / "lib" / lib_name;

    dll_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dll_handle_) {
      throw std::runtime_error("dlopen failed: " + lib_path.string() + " : " + std::string(dlerror()));
    }

    create_inferencer_ = load_symbol<CreateInferencerFunc>("createInferencer");
    delete_inferencer_ = load_symbol<DeleteInferencerFunc>("deleteInferencer");

    auto load_model_fn   = load_symbol<LoadModelFunc>("loadModel");
    auto get_input_fn    = load_symbol<GetInputBufferFunc>("getInputBuffer");
    auto get_output_fn   = load_symbol<GetOutputBufferFunc>("getOutputBuffer");
    auto infer_fn        = load_symbol<InferFunc>("infer");
    auto get_error_fn    = load_symbol<GetErrorStringFunc>("getErrorString");

    handle_ = create_inferencer_(nullptr);
    if (!handle_) {
      throw std::runtime_error("createInferencer returned null");
    }

    load_model_        = std::bind(load_model_fn, handle_, std::placeholders::_1);
    get_input_buffer_  = std::bind(get_input_fn, handle_, std::placeholders::_1, std::placeholders::_2);
    get_output_buffer_ = std::bind(get_output_fn, handle_, std::placeholders::_1, std::placeholders::_2);
    infer_             = std::bind(infer_fn, handle_);
    get_error_string_  = std::bind(get_error_fn, handle_);
  }

  Inferencer(const Inferencer&) = delete;
  Inferencer& operator=(const Inferencer&) = delete;

  ~Inferencer()
  {
    if (handle_ && delete_inferencer_) {
      delete_inferencer_(handle_);
      handle_ = nullptr;
    }
    if (dll_handle_) {
      dlclose(dll_handle_);
      dll_handle_ = nullptr;
    }
  }

  bool loadModel(const std::string & model_name) { return load_model_(model_name.c_str()); }

  std::size_t getInputBuffer(const std::string & name, void ** buf)
  {
    return static_cast<std::size_t>(get_input_buffer_(name.c_str(), buf));
  }

  std::size_t getOutputBuffer(const std::string & name, void ** buf)
  {
    return static_cast<std::size_t>(get_output_buffer_(name.c_str(), buf));
  }

  bool infer() { return infer_(); }

  std::string getErrorString()
  {
    const char * err = get_error_string_();
    return err ? std::string(err) : std::string();
  }

private:
  template <typename T>
  T load_symbol(const char * sym)
  {
    dlerror();  // clear
    void * p = dlsym(dll_handle_, sym);
    const char * e = dlerror();
    if (e || !p) {
      throw std::runtime_error(std::string("dlsym failed for ") + sym + " : " + (e ? e : "null"));
    }
    return reinterpret_cast<T>(p);
  }

  using CreateInferencerFunc = void * (*)(void *);
  using DeleteInferencerFunc = void (*)(void *);
  using LoadModelFunc        = bool (*)(void *, const char *);
  using GetInputBufferFunc   = unsigned (*)(void *, const char *, void **);
  using GetOutputBufferFunc  = unsigned (*)(void *, const char *, void **);
  using InferFunc            = bool (*)(void *);
  using GetErrorStringFunc   = const char * (*)(void *);

  void * dll_handle_ = nullptr;
  void * handle_ = nullptr;

  CreateInferencerFunc create_inferencer_ = nullptr;
  DeleteInferencerFunc delete_inferencer_ = nullptr;

  std::function<bool(const char *)> load_model_;
  std::function<unsigned(const char *, void **)> get_input_buffer_;
  std::function<unsigned(const char *, void **)> get_output_buffer_;
  std::function<bool()> infer_;
  std::function<const char *()> get_error_string_;
};
}  // namespace inferencer
