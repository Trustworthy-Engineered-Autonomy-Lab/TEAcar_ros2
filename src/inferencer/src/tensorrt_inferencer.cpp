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
    RTInferencer() : buffers(1, nullptr)
    {
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to create cuda stream: " + std::string(cudaGetErrorString(error)));
        }
    }

    ~RTInferencer()
    {
        cudaStreamDestroy(stream);
        for (const auto & buffer : buffers)
        {
            if (buffer != nullptr)
                cudaFree(buffer);
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
            {
                engine = loadEngine(engineFilePath.string());
            }

            if (engine == nullptr)
            {
                engine = loadOnnx(filePath.string());
                if (engine != nullptr)
                    saveEngine(engineFilePath.string());
            }
        }
        else if (fileExt == ".engine")
        {
            engine = loadEngine(filePath.string());
        }
        else
        {
            errorString = "Unsupported model file format " + fileExt;
            return false;
        }

        if (engine == nullptr)
        {
            return false;
        }

        context.reset(engine->createExecutionContext());
        if (context == nullptr)
        {
            throw std::runtime_error("Failed to create tensorrt execution context");
        }

        return true;
    }

    unsigned getInputBuffer(const std::string & inputName, void ** buffer) final
    {
        int inputIndex = engine->getBindingIndex(inputName.c_str());
        if (inputIndex == -1)
        {
            errorString = "Invalid input tensor name " + inputName;
            return 0;
        }

        nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
        nvinfer1::DataType inputDataType = engine->getBindingDataType(inputIndex);
        size_t inputSize = getVolume(inputDims) * getDataTypeSize(inputDataType);

        *buffer = allocBuffer(inputIndex, inputSize);
        if (*buffer == nullptr)
            return 0;

        return static_cast<unsigned>(inputSize);
    }

    unsigned getOutputBuffer(const std::string & outputName, void ** buffer) final
    {
        int outputIndex = engine->getBindingIndex(outputName.c_str());
        if (outputIndex == -1)
        {
            errorString = "Invalid output tensor name " + outputName;
            return 0;
        }

        nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
        nvinfer1::DataType outputDataType = engine->getBindingDataType(outputIndex);
        size_t outputSize = getVolume(outputDims) * getDataTypeSize(outputDataType);

        *buffer = allocBuffer(outputIndex, outputSize);
        if (*buffer == nullptr)
            return 0;

        return static_cast<unsigned>(outputSize);
    }

    bool infer() final
    {
        if (!context->enqueueV2(buffers.data(), stream, nullptr))
        {
            errorString = "Failed to enqueue the stream";
            return false;
        }

        cudaStreamSynchronize(stream);
        return true;
    }

    const std::string & getErrorString() const final
    {
        return errorString;
    }

    static std::shared_ptr<RTInferencer> create()
    {
        return std::make_shared<RTInferencer>();
    }

private:
    // ---------------------------------------------------------------
    // TensorRT object lifetime management
    // ---------------------------------------------------------------
    struct NvInferDeleter
    {
        template<typename T>
        void operator()(T * obj) const
        {
            if (obj)
                obj->destroy();
        }
    };

    // ---------------------------------------------------------------
    // Logger: replaces ROS_* macros with stderr output.
    // The inferencer is a shared library with no rclcpp node context,
    // so we write directly to stderr (consistent with dummy_inferencer).
    // ---------------------------------------------------------------
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char * msg) noexcept override
        {
            switch (severity)
            {
                case Severity::kVERBOSE:
                    fprintf(stderr, "[TRT DEBUG]   %s\n", msg);
                    break;
                case Severity::kINFO:
                    fprintf(stderr, "[TRT INFO]    %s\n", msg);
                    break;
                case Severity::kWARNING:
                    fprintf(stderr, "[TRT WARNING] %s\n", msg);
                    break;
                case Severity::kERROR:
                    fprintf(stderr, "[TRT ERROR]   %s\n", msg);
                    break;
                case Severity::kINTERNAL_ERROR:
                    fprintf(stderr, "[TRT FATAL]   %s\n", msg);
                    break;
                default:
                    fprintf(stderr, "[TRT UNKNOWN] %s\n", msg);
                    break;
            }
        }
    } logger;

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter>  context;

    std::vector<void *> buffers;
    cudaStream_t        stream;
    std::string         errorString;

    // ---------------------------------------------------------------
    // Build engine from ONNX
    // ---------------------------------------------------------------
    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
    loadOnnx(const std::string & fileName)
    {
        std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter>
            builder{nvinfer1::createInferBuilder(logger)};
        if (!builder)
            throw std::runtime_error("Failed to create tensorrt network builder");

        const auto explicitBatch =
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter>
            network{builder->createNetworkV2(explicitBatch)};
        if (!network)
            throw std::runtime_error("Failed to build tensorrt network");

        std::unique_ptr<nvonnxparser::IParser, NvInferDeleter>
            parser{nvonnxparser::createParser(*network, logger)};
        if (!parser)
            throw std::runtime_error("Failed to create onnx model parser");

        if (!parser->parseFromFile(
                fileName.c_str(),
                static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE)))
        {
            errorString = "Failed to parse onnx model file " + fileName;
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

        config->setMaxWorkspaceSize(totalMemory / 4);

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

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
            eng{builder->buildEngineWithConfig(*network, *config)};
        if (!eng)
        {
            errorString = "Failed to build tensorrt engine from onnx model " + fileName;
            return nullptr;
        }

        return eng;
    }

    // ---------------------------------------------------------------
    // Deserialise a .engine file
    // ---------------------------------------------------------------
    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
    loadEngine(const std::string & fileName)
    {
        std::ifstream file(fileName, std::ios::binary);
        if (!file)
        {
            errorString = "Model file " + fileName + " does not exist";
            return nullptr;
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(fileSize);
        file.read(engineData.data(), static_cast<std::streamsize>(fileSize));
        if (!file)
        {
            errorString = "Failed to read the model file " + fileName;
            return nullptr;
        }
        file.close();

        std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter>
            runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime)
            throw std::runtime_error("Failed to create tensorrt runtime");

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>
            eng{runtime->deserializeCudaEngine(engineData.data(), engineData.size())};
        if (!eng)
        {
            errorString = "Failed to deserialize tensorrt engine file " + fileName;
            return nullptr;
        }

        return eng;
    }

    // ---------------------------------------------------------------
    // Serialise engine to disk
    // ---------------------------------------------------------------
    bool saveEngine(const std::string & fileName)
    {
        std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter>
            serializedModel{engine->serialize()};

        std::ofstream file(fileName, std::ios::binary);
        if (!file)
        {
            errorString = "Failed to create the engine file " + fileName;
            return false;
        }

        file.write(
            static_cast<const char *>(serializedModel->data()),
            static_cast<std::streamsize>(serializedModel->size()));
        if (!file)
        {
            errorString = "Failed to write to the engine file " + fileName;
            return false;
        }

        file.close();
        return true;
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------
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
            if (dims.d[i] == -1)
                return 0;
            volume *= static_cast<size_t>(dims.d[i]);
        }
        return volume;
    }

    void * allocBuffer(int index, size_t size)
    {
        if (index + 1 > static_cast<int>(buffers.size()))
            buffers.resize(static_cast<size_t>(index) + 1, nullptr);

        if (buffers[index] != nullptr)
            return buffers[index];

        void * buffer = nullptr;
        cudaError_t error = cudaMallocManaged(&buffer, size);
        if (error != cudaSuccess)
            throw std::runtime_error(
                "Failed to allocate memory for tensor buffer: " +
                std::string(cudaGetErrorString(error)));

        buffers[index] = buffer;
        return buffer;
    }
};

}  // namespace inferencer

BOOST_DLL_ALIAS(inferencer::RTInferencer::create, tensorrt_inferencer)
