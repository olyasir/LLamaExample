#ifndef LLAMA_ENGINE_H
#define LLAMA_ENGINE_H

#include <string>
#include <vector>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <cpp/serve/config.h>
#include <cpp/serve/request.h>
#include <cpp/serve/engine.h>
#include "../include/engine_with_load.h"

using namespace mlc::llm;
using namespace mlc::llm::serve;


struct LLamaEngineCreationOutput {
  std::unique_ptr<EngineWithLoad> reloaded_engine;
  EngineConfig completed_engine_config;
  GenerationConfig default_generation_cfg;
  //ModelMetadata model_metadata;
};

class LlamaEngine : public Engine {
public:
    static Result<LLamaEngineCreationOutput> Create(const std::string& engine_config_json_str,
                                            Device device,
                                            FRequestStreamCallback request_stream_callback,
                                            Optional<EventTraceRecorder> trace_recorder) ;


                                            



   
};

#endif // LLAMA_ENGINE_H
