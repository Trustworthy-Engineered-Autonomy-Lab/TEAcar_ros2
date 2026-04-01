// tensorrt_inferencer.cpp (TensorRT 10 backend, ROS 2 compatible)
// Implements the Inferencer abstract base class using TensorRT 10 APIs.

#include <inferencer/inferencer.h>

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

#include <boost/dll/alias.hpp>

namespace inferencer
{

class RTInferencer : public Inferencer
{
public:
    RTInferencer()
    {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~RTInferencer()
    {
        cudaStreamDestroy(stream);
        for (auto& [name, buf] : buffers) {
            if (buf) cudaFree(buf);
        }
    }

    bool loadModel(const std::string& name) override
    {
        std::filesystem::path p(name);
        if (p.extension() == ".engine") {
            engine = loadEngine(name);
        } else if (p.extension() == ".onnx") {
            auto cached = p;
            cached.replace_extension(".engine");
            if (std::filesystem::exists(cached)) {
                engine = loadEngine(cached);
            }
            if (!engine) {
                engine = loadOnnx(name);
                if (engine) saveEngine(cached);
            }
        } else {
            errorString = "Unsupported model format";
            return false;
        }
        if (!engine) return false;
        context.reset(engine->createExecutionContext());
        return context != nullptr;
    }

    unsigned getInputBuffer(const std::string& name, void** buffer) override
    {
        // TRT 10: use tensor name-based API instead of binding index
        auto dims  = engine->getTensorShape(name.c_str());
        auto dtype = engine->getTensorDataType(name.c_str());
        if (dims.nbDims == 0) {
            errorString = "Tensor not found: " + name;
            return 0;
        }
        size_t sz = getVolume(dims) * getTypeSize(dtype);
        *buffer = allocBuffer(name, sz);
        return sz;
    }

    unsigned getOutputBuffer(const std::string& name, void** buffer) override
    {
        // TRT 10: use tensor name-based API instead of binding index
        auto dims  = engine->getTensorShape(name.c_str());
        auto dtype = engine->getTensorDataType(name.c_str());
        if (dims.nbDims == 0) {
            errorString = "Tensor not found: " + name;
            return 0;
        }
        size_t sz = getVolume(dims) * getTypeSize(dtype);
        *buffer = allocBuffer(name, sz);
        return sz;
    }

    bool infer() override
    {
        // TRT 10: set tensor addresses before enqueueV3
        for (auto& [name, ptr] : buffers) {
            if (!context->setTensorAddress(name.c_str(), ptr)) {
                errorString = "Failed to set tensor address for: " + name;
                return false;
            }
        }

        if (!context->enqueueV3(stream)) {
            errorString = "Failed to enqueue inference";
            return false;
        }

        cudaStreamSynchronize(stream);
        return true;
    }

    const std::string& getErrorString() const override
    {
        return errorString;
    }

    static std::shared_ptr<RTInferencer> create()
    {
        return std::make_shared<RTInferencer>();
    }

private:
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

    // ----- Helper: compute element count from dims -----
    static size_t getVolume(const nvinfer1::Dims& dims) {
        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) vol *= dims.d[i];
        return vol;
    }

    static size_t getTypeSize(nvinfer1::DataType type) {
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
    std::unique_ptr<nvinfer1::ICudaEngine> loadOnnx(const std::string& file) {
        // TRT 10: createNetworkV2(0) — kEXPLICIT_BATCH is deprecated / default
        auto builder = std::unique_ptr<nvinfer1::IBuilder>{nvinfer1::createInferBuilder(logger)};
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{
            builder->createNetworkV2(1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
        };
        auto parser  = std::unique_ptr<nvonnxparser::IParser>{nvonnxparser::createParser(*network, logger)};

        if (!parser->parseFromFile(file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            errorString = "Failed to parse ONNX: " + file;
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
            errorString = "Failed to build serialized network from: " + file;
            return nullptr;
        }

        auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(logger)};
        return std::unique_ptr<nvinfer1::ICudaEngine>{
            runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
    }

    // ----- Load a pre-built .engine file -----
    std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return nullptr;
        f.seekg(0, std::ios::end);
        size_t size = f.tellg();
        f.seekg(0);
        std::vector<char> buf(size);
        f.read(buf.data(), size);
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(logger)};
        return std::unique_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(buf.data(), size)};
    }

    // ----- Save serialized engine to disk -----
    bool saveEngine(const std::string& path) {
        auto ser = std::unique_ptr<nvinfer1::IHostMemory>{engine->serialize()};
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(static_cast<const char*>(ser->data()), ser->size());
        return true;
    }

    // ----- Allocate or return existing CUDA managed buffer for a tensor -----
    void* allocBuffer(const std::string& name, size_t sz) {
        auto it = buffers.find(name);
        if (it != buffers.end() && it->second != nullptr) {
            return it->second;
        }
        void* ptr = nullptr;
        cudaMallocManaged(&ptr, sz);
        buffers[name] = ptr;
        buf_sizes[name] = sz;
        return ptr;
    }
};

} // namespace inferencer

BOOST_DLL_ALIAS(inferencer::RTInferencer::create, tensorrt_inferencer);