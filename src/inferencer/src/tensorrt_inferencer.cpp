#include <cuda_runtime_api.h>
#include <inferencer/inferencer.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <boost/filesystem.hpp>
#include <boost/dll/alias.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace inferencer
{

class RTInferencer : public Inferencer
{
public:
    RTInferencer()
    {
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to create cuda stream: " + std::string(cudaGetErrorString(error)));
        }
    }

    ~RTInferencer()
    {
        cudaStreamDestroy(stream_);
        for (auto & kv : named_buffers_)
        {
            if (kv.second != nullptr)
                cudaFree(kv.second);
        }
    }

    bool loadModel(const std::string & modelName) final
    {
        boost::filesystem::path filePath(modelName);
        std::string fileExt = filePath.extension().string();

        if (fileExt == ".onnx")
        {
            boost::filesystem::path engineFilePath = filePath;
            engineFilePath.replace_extension("engine");

            if (boost::filesystem::exists(engineFilePath))
                engine_ = loadEngine(engineFilePath.string());

            if (engine_ == nullptr)
            {
                engine_ = loadOnnx(filePath.string());
                if (engine_ != nullptr)
                    saveEngine(engineFilePath.string());
            }
        }
        else if (fileExt == ".engine")
        {
            engine_ = loadEngine(filePath.string());
        }
        else
        {
            errorString_ = "Unsupported model file format " + fileExt;
            return false;
        }

        if (engine_ == nullptr)
            return false;

        context_.reset(engine_->createExecutionContext());
        if (context_ == nullptr)
            throw std::runtime_error("Failed to create tensorrt execution context");

        return true;
    }

    // -----------------------------------------------------------------
    // TRT 9 name-based tensor API replaces index-based binding API [1]
    // -----------------------------------------------------------------
    unsigned getInputBuffer(const std::string & inputName, void ** buffer) final
    {
        if (!engine_->isShapeInferenceIO(inputName.c_str()) &&
            engine_->getTensorIOMode(inputName.c_str()) != nvinfer1::TensorIOMode::kINPUT)
        {
            errorString_ = "Invalid input tensor name " + inputName;
            return 0;
        }

        nvinfer1::Dims dims     = engine_->getTensorShape(inputName.c_str());
        nvinfer1::DataType type = engine_->getTensorDataType(inputName.c_str());
        size_t size             = getVolume(dims) * getDataTypeSize(type);

        *buffer = allocNamedBuffer(inputName, size);
        if (*buffer == nullptr)
            return 0;

        // Tell the execution context where this tensor lives [1]
        context_->setTensorAddress(inputName.c_str(), *buffer);
        return static_cast<unsigned>(size);
    }

    unsigned getOutputBuffer(const std::string & outputName, void ** buffer) final
    {
        if (engine_->getTensorIOMode(outputName.c_str()) != nvinfer1::TensorIOMode::kOUTPUT)
        {
            errorString_ = "Invalid output tensor name " + outputName;
            return 0;
        }

        nvinfer1::Dims dims     = engine_->getTensorShape(outputName.c_str());
        nvinfer1::DataType type = engine_->getTensorDataType(outputName.c_str());
        size_t size             = getVolume(dims) * getDataTypeSize(type);

        *buffer = allocNamedBuffer(outputName, size);
        if (*buffer == nullptr)
            return 0;

        context_->setTensorAddress(outputName.c_str(), *buffer);
        return static_cast<unsigned>(size);
    }

    // -----------------------------------------------------------------
    // enqueueV2 removed in TRT 9 — use enqueueV3 [2]
    // enqueueV3 uses addresses set via setTensorAddress above,
    // so no buffer pointer array is needed here.
    // -----------------------------------------------------------------
    bool infer() final
    {
        if (!context_->enqueueV3(stream_))
        {
            errorString_ = "Failed to enqueue the stream";
            return false;
        }
        cudaStreamSynchronize(stream_);
        return true;
    }

    const std::string & getErrorString() const final
    {
        return errorString_;
    }

    static std::shared_ptr<RTInferencer> create()
    {
        return std::make_shared<RTInferencer>();
    }

private:
    // -----------------------------------------------------------------
    // TRT 9 removed destroy() — objects are now plain delete [3]
    // -----------------------------------------------------------------
    struct NvInferDeleter
    {
        template<typename T>
        void operator()(T * obj) const { delete obj; }
    };

    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char * msg) noexcept override
        {
            switch (severity)
            {
                case Severity::kVERBOSE:        fprintf(stderr, "[TRT DEBUG]   %s\n", msg); break;
                case Severity::kINFO:           fprintf(stderr, "[TRT INFO]    %s\n", msg); break;
                case Severity::kWARNING:        fprintf(stderr, "[TRT WARNING] %s\n", msg); break;
                case Severity::kERROR:          fprintf(stderr, "[TRT ERROR]   %s\n", msg); break;
                case Severity::kINTERNAL_ERROR: fprintf(stderr, "[TRT FATAL]   %s\n", msg); break;
                default:                        fprintf(stderr, "[TRT UNKNOWN] %s\n", msg); break;
            }
        }
    } logger_;

    std::unique_ptr<nvinfer1::ICudaEngine,      NvInferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> context_;

    // Named buffer map — keyed by tensor name
    std::unordered_map<std::string, void *> named_buffers_;
    cudaStream_t stream_;
    std::string  errorString_;

    // -----------------------------------------------------------------
    // Build engine from ONNX
    // kEXPLICIT_BATCH deprecated in TRT 8, use createNetworkV2(0) in TRT 9 [1]
    // setMaxWorkspaceSize removed — use setMemoryPoolLimit [1]
    // buildEngineWithConfig removed — use buildSerializedNetwork [1]
    // -----------------------------------------------------------------
    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
    loadOnnx(const std::string & fileName)
    {
        std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter>
            builder{nvinfer1::createInferBuilder(logger_)};
        if (!builder)
            throw std::runtime_error("Failed to create tensorrt network builder");

        // Pass 0 — kEXPLICIT_BATCH is the only mode and is now the default [1]
        std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter>
            network{builder->createNetworkV2(0)};
        if (!network)
            throw std::runtime_error("Failed to build tensorrt network");

        std::unique_ptr<nvonnxparser::IParser, NvInferDeleter>
            parser{nvonnxparser::createParser(*network, logger_)};
        if (!parser)
            throw std::runtime_error("Failed to create onnx model parser");

        if (!parser->parseFromFile(
                fileName.c_str(),
                static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE)))
        {
            errorString_ = "Failed to parse onnx model file " + fileName;
            return nullptr;
        }

        std::unique_ptr<nvinfer1::IBuilderConfig, NvInferDeleter>
            config{builder->createBuilderConfig()};
        if (!config)
            throw std::runtime_error("Failed to create tensorrt builder configuration");

        size_t totalMemory = 0;
        cudaError_t err = cudaMemGetInfo(nullptr, &totalMemory);
        if (err != cudaSuccess)
            throw std::runtime_error(
                "Failed to get cuda memory info: " + std::string(cudaGetErrorString(err)));

        // setMaxWorkspaceSize removed in TRT 9 [1]
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, totalMemory / 4);

        nvinfer1::IOptimizationProfile * profile = builder->createOptimizationProfile();
        if (!profile)
            throw std::runtime_error("Failed to create tensorrt optimization profile");

        nvinfer1::ITensor * input     = network->getInput(0);
        const char *        inputName = input->getName();
        nvinfer1::Dims      dims      = input->getDimensions();
        dims.d[0] = 1;

        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
        config->addOptimizationProfile(profile);

        // buildEngineWithConfig removed in TRT 9 — serialize then deserialize [1]
        std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter>
            serialized{builder->buildSerializedNetwork(*network, *config)};
        if (!serialized)
        {
            errorString_ = "Failed to build serialized network from " + fileName;
            return nullptr;
        }

        std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter>
            runtime{nvinfer1::createInferRuntime(logger_)};
        if (!runtime)
            throw std::runtime_error("Failed to create tensorrt runtime");

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
            eng{runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
        if (!eng)
        {
            errorString_ = "Failed to deserialize engine built from " + fileName;
            return nullptr;
        }

        return eng;
    }

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
    loadEngine(const std::string & fileName)
    {
        std::ifstream file(fileName, std::ios::binary);
        if (!file)
        {
            errorString_ = "Model file " + fileName + " does not exist";
            return nullptr;
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(fileSize);
        file.read(engineData.data(), static_cast<std::streamsize>(fileSize));
        if (!file)
        {
            errorString_ = "Failed to read the model file " + fileName;
            return nullptr;
        }
        file.close();

        std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter>
            runtime{nvinfer1::createInferRuntime(logger_)};
        if (!runtime)
            throw std::runtime_error("Failed to create tensorrt runtime");

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
            eng{runtime->deserializeCudaEngine(engineData.data(), engineData.size())};
        if (!eng)
        {
            errorString_ = "Failed to deserialize tensorrt engine file " + fileName;
            return nullptr;
        }

        return eng;
    }

    bool saveEngine(const std::string & fileName)
    {
        std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter>
            serialized{engine_->serialize()};

        std::ofstream file(fileName, std::ios::binary);
        if (!file)
        {
            errorString_ = "Failed to create the engine file " + fileName;
            return false;
        }

        file.write(
            static_cast<const char *>(serialized->data()),
            static_cast<std::streamsize>(serialized->size()));
        if (!file)
        {
            errorString_ = "Failed to write to the engine file " + fileName;
            return false;
        }

        file.close();
        return true;
    }

    static size_t getDataTypeSize(nvinfer1::DataType type)
    {
        switch (type)
        {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF:  return 2;
            case nvinfer1::DataType::kINT8:  return 1;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kBOOL:  return 1;
            default:                         return 0;
        }
    }

    static size_t getVolume(const nvinfer1::Dims & dims)
    {
        size_t volume = 1;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] == -1) return 0;
            volume *= static_cast<size_t>(dims.d[i]);
        }
        return volume;
    }

    void * allocNamedBuffer(const std::string & name, size_t size)
    {
        auto it = named_buffers_.find(name);
        if (it != named_buffers_.end() && it->second != nullptr)
            return it->second;

        void * buffer = nullptr;
        cudaError_t error = cudaMallocManaged(&buffer, size);
        if (error != cudaSuccess)
            throw std::runtime_error(
                "Failed to allocate memory for tensor '" + name + "': " +
                std::string(cudaGetErrorString(error)));

        named_buffers_[name] = buffer;
        return buffer;
    }
};

}  // namespace inferencer

BOOST_DLL_ALIAS(inferencer::RTInferencer::create, tensorrt_inferencer)
