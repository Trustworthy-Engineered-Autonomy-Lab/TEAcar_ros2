// tensorrt_inferencer.cpp (TensorRT 10 backend, ROS 2 compatible, no ROS logging)
// Implements the C API defined in inferencer/inferencer_c.h

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <filesystem>

// ---------------------------------------------------------------------------
// TRT 10 uses standard delete instead of the old destroy() method.
// ---------------------------------------------------------------------------

struct RTInferencer
{
    RTInferencer() {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~RTInferencer() {
        cudaStreamDestroy(stream);
        for (auto& [name, buf] : buffers) {
            if (buf) cudaFree(buf);
        }
    }

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TensorRT] " << msg << std::endl;
            }
        }
    } logger;

    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::map<std::string, void*> buffers;    // tensor name -> GPU buffer
    std::map<std::string, size_t> buf_sizes; // tensor name -> byte size
    cudaStream_t stream;
    std::string errorString;
};

namespace
{

// ----- Helper: compute element count from dims -----
size_t getVolume(const nvinfer1::Dims& dims) {
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) vol *= dims.d[i];
    return vol;
}

size_t getTypeSize(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    default: return 0;
    }
}

// ----- Load from ONNX, build serialized network, then deserialize -----
std::unique_ptr<nvinfer1::ICudaEngine> loadOnnx(RTInferencer* infer, const std::string& file) {
    // TRT 10: createNetworkV2(0) — kEXPLICIT_BATCH is deprecated / default
    auto builder = std::unique_ptr<nvinfer1::IBuilder>{nvinfer1::createInferBuilder(infer->logger)};
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{builder->createNetworkV2(0)};
    auto parser  = std::unique_ptr<nvonnxparser::IParser>{nvonnxparser::createParser(*network, infer->logger)};

    if (!parser->parseFromFile(file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        infer->errorString = "Failed to parse ONNX: " + file;
        return nullptr;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};
    size_t totalMem = 0;
    cudaMemGetInfo(nullptr, &totalMem);
    // TRT 10: setMemoryPoolLimit replaces setMaxWorkspaceSize
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, totalMem / 4);

    // TRT 10: buildSerializedNetwork replaces buildEngineWithConfig
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) {
        infer->errorString = "Failed to build serialized network from: " + file;
        return nullptr;
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(infer->logger)};
    return std::unique_ptr<nvinfer1::ICudaEngine>{
        runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
}

// ----- Load a pre-built .engine file -----
std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(RTInferencer* infer, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(infer->logger)};
    return std::unique_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(buf.data(), size)};
}

// ----- Save serialized engine to disk -----
bool saveEngine(RTInferencer* infer, const std::string& path) {
    auto ser = std::unique_ptr<nvinfer1::IHostMemory>{infer->engine->serialize()};
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(static_cast<const char*>(ser->data()), ser->size());
    return true;
}

// ----- Allocate or return existing CUDA managed buffer for a tensor -----
void* allocBuffer(RTInferencer* inf, const std::string& name, size_t sz) {
    auto it = inf->buffers.find(name);
    if (it != inf->buffers.end() && it->second != nullptr) {
        return it->second;
    }
    void* ptr = nullptr;
    cudaMallocManaged(&ptr, sz);
    inf->buffers[name] = ptr;
    inf->buf_sizes[name] = sz;
    return ptr;
}

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------

extern "C" void* createInferencer(void*) {
    return new RTInferencer();
}

extern "C" void deleteInferencer(void* handle) {
    delete static_cast<RTInferencer*>(handle);
}

extern "C" bool loadModel(void* h, const char* name) {
    auto* i = static_cast<RTInferencer*>(h);
    std::filesystem::path p(name);
    if (p.extension() == ".engine") {
        i->engine = loadEngine(i, name);
    } else if (p.extension() == ".onnx") {
        auto cached = p;
        cached.replace_extension(".engine");
        if (std::filesystem::exists(cached)) {
            i->engine = loadEngine(i, cached);
        }
        if (!i->engine) {
            i->engine = loadOnnx(i, name);
            if (i->engine) saveEngine(i, cached);
        }
    } else {
        i->errorString = "Unsupported model format";
        return false;
    }
    if (!i->engine) return false;
    i->context.reset(i->engine->createExecutionContext());
    return i->context != nullptr;
}

extern "C" unsigned getInputBuffer(void* h, const char* name, void** buf) {
    auto* i = static_cast<RTInferencer*>(h);
    // TRT 10: use tensor name-based API instead of binding index
    auto dims  = i->engine->getTensorShape(name);
    auto dtype = i->engine->getTensorDataType(name);
    if (dims.nbDims == 0) {
        i->errorString = std::string("Tensor not found: ") + name;
        return 0;
    }
    size_t sz = getVolume(dims) * getTypeSize(dtype);
    *buf = allocBuffer(i, name, sz);
    return sz;
}

extern "C" unsigned getOutputBuffer(void* h, const char* name, void** buf) {
    auto* i = static_cast<RTInferencer*>(h);
    // TRT 10: use tensor name-based API instead of binding index
    auto dims  = i->engine->getTensorShape(name);
    auto dtype = i->engine->getTensorDataType(name);
    if (dims.nbDims == 0) {
        i->errorString = std::string("Tensor not found: ") + name;
        return 0;
    }
    size_t sz = getVolume(dims) * getTypeSize(dtype);
    *buf = allocBuffer(i, name, sz);
    return sz;
}

extern "C" bool infer(void* h) {
    auto* i = static_cast<RTInferencer*>(h);

    // TRT 10: set tensor addresses before enqueueV3
    for (auto& [name, ptr] : i->buffers) {
        if (!i->context->setTensorAddress(name.c_str(), ptr)) {
            i->errorString = "Failed to set tensor address for: " + name;
            return false;
        }
    }

    if (!i->context->enqueueV3(i->stream)) {
        i->errorString = "Failed to enqueue inference";
        return false;
    }

    cudaStreamSynchronize(i->stream);
    return true;
}

extern "C" const char* getErrorString(void* h) {
    return static_cast<RTInferencer*>(h)->errorString.c_str();
}

} // anonymous namespace