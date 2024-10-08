#include "../include/llama_model.h"
#include "../include/logging.h"
#include<cpp/serve/model.h>
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

using namespace mlc::llm;
using namespace mlc::llm::serve;

loglevel_e loglevel = logDEBUG;

void  callback_func(Array<RequestStreamOutput> out){}

std::string  get_engine_config(String model_path, String model_lib)
{
    std::string engine_config = "{\"model\" : \"" + model_path+"\"";
    engine_config = engine_config+ ", \"model_lib\" : \""+model_lib+"\"";
    engine_config += ", \"additional_models\" : [] , \"mode\" : \"interactive\", \"tensor_parallel_shards\" : null,\
                        \"pipeline_parallel_stages\" : null, \"gpu_memory_utilization\" : null,\
                         \"kv_cache_page_size\" : 16, \"max_num_sequence\" : null, \"max_total_sequence_length\" : null,\
                         \"max_single_sequence_length\" : null, \"prefill_chunk_size\" : null, \"sliding_window_size\" : null,\
                         \"attention_sink_size\" : null, \"max_history_size\" : null, \"kv_state_kind\" : null,\
                         \"speculative_mode\" : \"disable\", \"spec_draft_length\" : 0, \"spec_tree_width\" : 1,\
                         \"prefix_cache_mode\" : \"radix\", \"prefix_cache_max_num_recycling_seqs\" : null,\
                         \"prefill_mode\" : \"hybrid\", \"verbose\" : false}";
    return engine_config;
}


 LlamaModel::LlamaModel(String model_path, String model_lib, std::string generation_config_str ) {
     log(logDEBUG) << "Model Path "    << model_path << " \n";
     log(logDEBUG) << "model_lib "    << model_lib << " \n";
     std::string engine_config = get_engine_config(model_path, model_lib);
     tokenizer = Tokenizer::FromPath(model_path);
     log(logDEBUG) << "Loaded Tokenizer..\n";
     
     log(logDEBUG) << "Started Engine Creation..\n";
     Result<EngineCreationOutput> output_res = Engine::Create(
         engine_config, tvm::Device{ static_cast<DLDeviceType>(kDLVulkan), 0 },
         callback_func, {});
     log(logDEBUG) << "Finished Engine creation..\n";
     EngineCreationOutput output = output_res.Unwrap(); 
     GenerationConfig default_generation_config  = output.default_generation_cfg;
     auto additions_generation_config = json::ParseToJSONObject(generation_config_str);
     auto combined_config_res = GenerationConfig::FromJSON(additions_generation_config, default_generation_config);
    _generation_config = combined_config_res.Unwrap();
     _engine = std::move(output.reloaded_engine);   
     _model_path = model_path;
     log(logDEBUG) << "Finished model initialization\n";
 }

 bool LlamaModel::LoadWeights(const std::string& weights_file) {
     return true;
 }

 // Generate texts based on input prompts
 std::string LlamaModel::Process(const std::string prompt) {
        log(logDEBUG) << "Start Processing prompt"<<prompt<<"\n";
         //std::vector<std::optional<std::vector<std::vector<std::string>>>> output_logprobs_str(prompts.size());
         size_t num_finished_generations = 0;
         int num_total_generations = 1;
         std::string output_texts;
         bool finished_generation = false;
         TextStreamer streamer = TextStreamer(tokenizer);
         auto lambda = [&](Array<RequestStreamOutput> out) {
             for (const auto& delta_output : out) {
                 std::string request_id = delta_output->request_id;
                 std::vector<std::vector<int64_t>> group_delta_token_ids = delta_output->group_delta_token_ids;
                 std::vector<Optional<String>> group_finish_reason = delta_output->group_finish_reason;
                 if (delta_output->group_extra_prefix_string.size() > 0)
                 {
                     output_texts.append(delta_output->group_extra_prefix_string[0]);
                 }
                 if (delta_output->group_delta_token_ids.size() > 0) {

                     std::vector<int64_t > delta_token_ids = delta_output->group_delta_token_ids[0];
                     std::vector<int32_t>  ids;
                     for (auto i : delta_token_ids) {
                         ids.push_back(i);
                     }
                     std::string decoded = streamer->Put(ids);
                     output_texts.append(decoded);
                 }
                 if (delta_output->group_finish_reason.size() > 0)
                 {
                     Optional<String> finish_reason = delta_output->group_finish_reason[0];
                     if (finish_reason.defined()) {
                         std::string val = finish_reason.value();
                         output_texts.append(streamer->Finish());
                         finished_generation = true;
                         log(logDEBUG) << "Finish generating output, reason: "<<val<<"\n";
                     }
                 }
             }
             };
         // Override callback in the engine
         GetEngine()->SetRequestStreamCallback(lambda);
         tvm::runtime::String req_id = "0";
         auto request_data = TextData(String(prompt));
         tvm::runtime::Array<Data> inputs;
         inputs.push_back(request_data);
         Request req = Request(std::move(req_id), std::move(inputs), _generation_config);
         GetEngine()->AddRequest(req);

         // Process requests until generation is complete
         while (! finished_generation) {
             GetEngine()->Step();
         }
         return  output_texts;
 }



