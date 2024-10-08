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
    /**
     * Constructor for llama model
     *
     * @param model_path - path to a model directory that includes the weight shards and mlc-chat-config.json.
     * @param model_lib path to .so model file.
     * @param generation_config_str json string for generation configuration
     */
    LlamaModel(String model_path, String model_lib, std::string generation_config_str) ;

    //// Load model weights from a file
    bool LoadWeights(const std::string& weights_file);

    /**
     * The chat method, it reserves in input string and generated a reply.
     *
     * @param prompt - user request from the model in natural language.
     * @return a replay from the model
     */
    std::string Process(const std::string prompt);

private:
    std::unique_ptr<Engine> _engine = nullptr;
    GenerationConfig _generation_config;
    Tokenizer tokenizer;
    std::string _model_path;
   

    Engine* GetEngine() {
        if (_engine == nullptr) {
            std::cerr << "Engine is not initialized via init\n";
            assert(false);
            
        }
        return _engine.get();
    }

    
  
};

#endif // LLAMA_MODEL_H
