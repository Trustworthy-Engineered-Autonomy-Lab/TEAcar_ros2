// mock_inferencer.cpp
// A simple mock backend implementing inferencer_c.h symbols.
// - Input buffer: 224x144 RGB float32 => 224*144*3*4 = 387072 bytes
// - Output buffer: single float (4 bytes), used as "steer" estimate

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include "inferencer/inferencer_c.h"

struct MockInferencer
{
  std::vector<uint8_t> input_bytes;  // raw bytes for input tensor
  float output_value = 0.0f;         // 4-byte output (steer)
  std::string error;
  bool model_loaded = false;
};

static constexpr unsigned kW = 224;
static constexpr unsigned kH = 144;
static constexpr unsigned kC = 3;  // RGB
static constexpr unsigned kInputBytes = kW * kH * kC * 4; // float32
static constexpr unsigned kOutputBytes = sizeof(float);

extern "C" void* createInferencer(void* /*options*/)
{
  try {
    return new MockInferencer();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void deleteInferencer(void* inferencer)
{
  delete static_cast<MockInferencer*>(inferencer);
}

extern "C" bool loadModel(void* inferencer, const char* modelName)
{
  auto* m = static_cast<MockInferencer*>(inferencer);
  if (!m) return false;

  // This is a mock, accept any model path/name as "loaded".
  (void)modelName;
  m->model_loaded = true;
  m->error.clear();
  return true;
}

extern "C" unsigned getInputBuffer(void* inferencer, const char* /*inputName*/, void** buffer)
{
  auto* m = static_cast<MockInferencer*>(inferencer);
  if (!m || !buffer) return 0;

  if (!m->model_loaded) {
    m->error = "MockInferencer: loadModel() must be called before getInputBuffer()";
    *buffer = nullptr;
    return 0;
  }

  m->input_bytes.resize(kInputBytes);
  *buffer = m->input_bytes.data();
  return kInputBytes;
}

extern "C" unsigned getOutputBuffer(void* inferencer, const char* /*outputName*/, void** buffer)
{
  auto* m = static_cast<MockInferencer*>(inferencer);
  if (!m || !buffer) return 0;

  if (!m->model_loaded) {
    m->error = "MockInferencer: loadModel() must be called before getOutputBuffer()";
    *buffer = nullptr;
    return 0;
  }

  // Output is 4 bytes (float)
  *buffer = &(m->output_value);
  return kOutputBytes;
}

extern "C" bool infer(void* inferencer)
{
  auto* m = static_cast<MockInferencer*>(inferencer);
  if (!m) return false;

  if (!m->model_loaded) {
    m->error = "MockInferencer: loadModel() must be called before infer()";
    return false;
  }

  // Deterministic mock behavior:
  // produce a stable steer estimate = 0.0f
  // (we can later change this to something based on input_bytes)
  m->output_value = 0.0f;

  m->error.clear();
  return true;
}

extern "C" const char* getErrorString(void* inferencer)
{
  auto* m = static_cast<MockInferencer*>(inferencer);
  if (!m) return "MockInferencer: null handle";
  return m->error.c_str();
}
