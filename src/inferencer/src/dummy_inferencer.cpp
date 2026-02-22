// DUMMY: This entire file is a test-only inferencer backend.
// DUMMY: Remove before publication.
//
// dummy_inferencer.cpp — No-GPU inferencer backend for testing.
// Implements the C API defined in inferencer/inferencer_c.h.
// Always outputs steer = 0.0 (drive straight).

#include "inferencer/inferencer_c.h"
#include <cstring>
#include <string>

struct DummyInferencer
{
    float input_buffer[1 * 3 * 224 * 224]; // matches expected image tensor size
    float output_buffer[1];                // single steer value
    std::string errorString;
};

extern "C" {

void* createInferencer(void* /*options*/) {
    // DUMMY: allocate a trivial inferencer with no GPU
    return new DummyInferencer();
}

void deleteInferencer(void* inferencer) {
    delete static_cast<DummyInferencer*>(inferencer);
}

bool loadModel(void* /*inferencer*/, const char* /*modelName*/) {
    // DUMMY: accept any model file, do nothing
    return true;
}

unsigned getInputBuffer(void* inferencer, const char* /*inputName*/, void** buffer) {
    auto* d = static_cast<DummyInferencer*>(inferencer);
    *buffer = d->input_buffer;
    return sizeof(d->input_buffer);
}

unsigned getOutputBuffer(void* inferencer, const char* /*outputName*/, void** buffer) {
    auto* d = static_cast<DummyInferencer*>(inferencer);
    d->output_buffer[0] = 0.0f; // DUMMY: always steer straight
    *buffer = d->output_buffer;
    return sizeof(d->output_buffer);
}

bool infer(void* inferencer) {
    // DUMMY: "inference" just sets output to 0.0 (steer straight)
    auto* d = static_cast<DummyInferencer*>(inferencer);
    d->output_buffer[0] = 0.0f;
    return true;
}

const char* getErrorString(void* inferencer) {
    return static_cast<DummyInferencer*>(inferencer)->errorString.c_str();
}

} // extern "C"
