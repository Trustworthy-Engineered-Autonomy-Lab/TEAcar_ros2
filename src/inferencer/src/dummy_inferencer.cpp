// DUMMY: This entire file is a test-only inferencer backend.
// DUMMY: Remove before publication.
//
// dummy_inferencer.cpp — No-GPU inferencer backend for testing.
// Implements the Inferencer abstract base class.
// Generates sine-wave steering or turns on red detection.

#include <inferencer/inferencer.h>

#include <boost/dll/alias.hpp>

#include <cstring>
#include <string>
#include <chrono>
#include <cmath>
#include <memory>

namespace inferencer
{

class DummyInferencer : public Inferencer
{
public:
    DummyInferencer() = default;
    ~DummyInferencer() override = default;

    bool loadModel(const std::string& /*path*/) override
    {
        // DUMMY: accept any model file, do nothing
        return true;
    }

    unsigned getInputBuffer(const std::string& /*name*/, void** buffer) override
    {
        *buffer = input_buffer_;
        return sizeof(input_buffer_);
    }

    unsigned getOutputBuffer(const std::string& /*name*/, void** buffer) override
    {
        output_buffer_[0] = 0.0f; // DUMMY: always steer straight initially
        *buffer = output_buffer_;
        return sizeof(output_buffer_);
    }

    bool infer() override
    {
        // DUMMY COLOR DETECTION (very basic, for visual verification):
        // Check if the center chunk of the image is predominantly red.
        // The input buffer is assumed to be BGR format as per the GSCAM config,
        // though the dummy model python script uses [1,3,224,224] NCHW layout.
        // We'll just sample a few pixels to see if one channel is super high.
        bool detect_red = false;
        float sum_r = 0, sum_g = 0, sum_b = 0;
        // Sample a small patch in the middle. Assuming NCHW float32 [1, 3, 224, 224]
        int center_start = (224 * 110) + 110;
        for (int i = 0; i < 10; ++i) {
            sum_b += input_buffer_[0 * 224 * 224 + center_start + i]; // B
            sum_g += input_buffer_[1 * 224 * 224 + center_start + i]; // G
            sum_r += input_buffer_[2 * 224 * 224 + center_start + i]; // R
        }

        // If it's very red (e.g. holding a red object in front of the camera)
        if (sum_r > (sum_g + sum_b) * 1.5f && sum_r > 10.0f) {
            detect_red = true;
        }

        if (detect_red) {
            // Since we only control steer, output a sharp turn.
            output_buffer_[0] = 1.0f;
        } else {
            // DUMMY SINE WAVE STEERING:
            // Use the system clock to generate a sine wave between -1.0 and 1.0.
            // This makes the wheels turn side to side smoothly over time.
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            double seconds = std::chrono::duration<double>(now).count();

            // Sine wave with a 3-second period
            output_buffer_[0] = static_cast<float>(std::sin(seconds * 2.0 * 3.14159 / 3.0));
        }

        return true;
    }

    const std::string& getErrorString() const override
    {
        return errorString_;
    }

    static std::shared_ptr<DummyInferencer> create()
    {
        return std::make_shared<DummyInferencer>();
    }

private:
    float input_buffer_[1 * 3 * 224 * 224] = {}; // matches expected image tensor size
    float output_buffer_[1] = {};                 // single steer value
    std::string errorString_;
};

} // namespace inferencer

BOOST_DLL_ALIAS(inferencer::DummyInferencer::create, dummy_inferencer);
