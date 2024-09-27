#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

class LlamaModel {
public:
    // Constructor
    LlamaModel();

    // Load model weights from a file
    bool LoadWeights(const std::string& weights_file);

    // Process input and generate output
    std::vector<float> Process(const std::vector<float>& input);

protected:
  tvm::runtime::Module lib_;
  
};

#endif // LLAMA_MODEL_H
