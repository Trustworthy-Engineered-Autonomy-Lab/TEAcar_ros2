// DUMMY: This entire file is a test-only inferencer backend.
// DUMMY: Remove before publication.
//
// dummy_inferencer.cpp — No-GPU inferencer backend for testing.
// Implements the C API defined in inferencer/inferencer_c.h.
// Always outputs steer = 0.0 (drive straight).

#include "inferencer/inferencer_c.h"
#include <cstring>
#include <string>
#include <chrono>
#include <cmath>

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
    auto* d = static_cast<DummyInferencer*>(inferencer);
    
    // DUMMY COLOR DETECTION (very basic, for visual verification):
    // Check if the center chunk of the image is predominantly red.
    // The input buffer is assumed to be BGR format as per the GSCAM config,
    // though the dummy model python script uses [1,3,224,224] NCHW layout.
    // We'll just sample a few pixels to see if one channel is super high.
    bool detect_red = false;
    float sum_r = 0, sum_g = 0, sum_b = 0;
    // Sample a small patch in the middle. Assuming NCHW float32 [1, 3, 224, 224]
    int center_start = (224 * 110) + 110; 
    for(int i=0; i<10; ++i) {
        sum_b += d->input_buffer[0 * 224 * 224 + center_start + i]; // B
        sum_g += d->input_buffer[1 * 224 * 224 + center_start + i]; // G
        sum_r += d->input_buffer[2 * 224 * 224 + center_start + i]; // R
    }
    
    // If it's very red (e.g. holding a red object in front of the camera)
    if (sum_r > (sum_g + sum_b) * 1.5f && sum_r > 10.0f) {
        detect_red = true;
    }

    if (detect_red) {
        // Stop turning, just go straight (or we could output a special value 
        // if this controller handled throttle, but it only handles steer).
        // Since we only control steer, we'll just output a sharp turn.
        d->output_buffer[0] = 1.0f; 
    } else {
        // DUMMY SINE WAVE STEERING:
        // Use the system clock to generate a sine wave between -1.0 and 1.0.
        // This makes the wheels turn side to side smoothly over time.
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        double seconds = std::chrono::duration<double>(now).count();
        
        // Sine wave with a 3-second period
        d->output_buffer[0] = static_cast<float>(std::sin(seconds * 2.0 * 3.14159 / 3.0));
    }

    return true;
}

const char* getErrorString(void* inferencer) {
    return static_cast<DummyInferencer*>(inferencer)->errorString.c_str();
}

} // extern "C"
