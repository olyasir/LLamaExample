#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <cpp/serve/config.h>
#include <cpp/serve/request.h>
#include <../include/llama_engine.h>
#include "../include/logging.h"
#include "../include/utils.h"
#include <cpp/serve/model.h>
#include <vector>
#include <cpp/serve/data.h>
#include <cpp/support/result.h>
#include <cpp/support/json_parser.h>
#include <cpp/tokenizers/tokenizers.h>
#include <cpp/tokenizers/streamer.h>
#include <picojson.h>
#include <fstream>
#include <iostream>
#include <string>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>
#include <filesystem>

using namespace mlc::llm;
using namespace mlc::llm::serve;
using namespace tvm::runtime::relax_vm;

class LlamaModel {
public:
    /**
     * Constructor for llama model
     *
     * @param model_path - path to a model directory that includes the weight shards and mlc-chat-config.json.
     * @param model_lib path to .so model file.
     * @param generation_config_str json string for generation configuration
     * @param device device to run the model on
     */
    LlamaModel(String model_path, String model_lib, std::string generation_config_str, tvm::Device device) ;

    
    std::string Metrics();
    
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
    std::unique_ptr<EngineWithLoad> _engine = nullptr;
    GenerationConfig _generation_config;
    Tokenizer tokenizer;
    std::string _model_path;
    int _loaded_weight_num;
    int _required_weight_num;
    bool _initialized;
    NDArrayCacheMetadata ndarray_cache_metadata_; 
    tvm::Device _device;
   
   

    EngineWithLoad* GetEngine() {
        if (_engine == nullptr) {
            std::cerr << "Engine is not initialized via init\n";
            assert(false);
            
        }
        return _engine.get();
    }

    void  LoadParams();

    
  
};

#endif // LLAMA_MODEL_H
