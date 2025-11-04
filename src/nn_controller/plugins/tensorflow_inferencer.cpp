#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include "nn_controller/inferencer_api.hpp"

namespace { constexpr int W=224, H=144, C=3; }

class TfBackend : public nnc::Inferencer {
  std::vector<unsigned char> in_;
  float out_{0.f};
public:
  bool loadModel(const std::string&) override { return true; }
  unsigned getInputBuffer(const std::string&, void** buf) override {
    in_.resize(W*H*C*4); *buf = in_.data(); return (unsigned)in_.size();
  }
  unsigned getOutputBuffer(const std::string&, void** buf) override {
    *buf=&out_; return sizeof(float);
  }
  bool infer() override { static float t=0.f; t+=0.05f; out_=0.25f*std::sin(t); return true; }
  std::string getErrorString() const override { return ""; }
};

extern "C" std::shared_ptr<nnc::Inferencer> tensorflow_inferencer() {
  return std::make_shared<TfBackend>();
}
