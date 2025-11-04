#pragma once
#include <memory>
#include <string>

namespace nnc {
class Inferencer {
public:
  virtual ~Inferencer() = default;
  virtual bool loadModel(const std::string& path) = 0;
  virtual unsigned getInputBuffer (const std::string& name, void** buffer) = 0;
  virtual unsigned getOutputBuffer(const std::string& name, void** buffer) = 0;
  virtual bool infer() = 0;
  virtual std::string getErrorString() const = 0;
};
using InferencerPtr = std::shared_ptr<Inferencer>;
} // namespace nnc
