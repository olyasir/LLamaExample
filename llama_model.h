#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <cpp/serve/config.h>
#include <cpp/serve/request.h>
#include <cpp/serve/engine.h>

using namespace mlc::llm;
using namespace mlc::llm::serve;

class LlamaModel {
public:
    // Constructor
    LlamaModel();

    //// Load model weights from a file
    bool LoadWeights(const std::string& weights_file);

    // Process input and generate output
    std::string LlamaModel::Process(const std::string prompt);




private:
    std::unique_ptr<Engine> _engine = nullptr;
    GenerationConfig _generation_config;
   

    Engine* GetEngine() {
        if (_engine == nullptr) {
            std::cerr << "Engine is not initialized via init\n";
            assert(false);
            
        }
        return _engine.get();
    }

    
  
};

#endif // LLAMA_MODEL_H
