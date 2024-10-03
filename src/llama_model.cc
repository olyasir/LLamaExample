#include "../include/llama_model.h"
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

using namespace mlc::llm;
using namespace mlc::llm::serve;




Tokenizer tokenizer;
TextStreamer streamer;



void  callback_func(Array<RequestStreamOutput> out)
{
    
    for (const auto& delta_output : out) {    
    }
}

 LlamaModel::LlamaModel() { 

     String engine_config = "{\"model\": \"C:\\\\Users\\\\sirki\\\\teher_workspace\\\\converted_tiny_llama\", \"model_lib\" : \"C:\\\\Users\\\\sirki\\\\teher_workspace\\\\tiny_llama_vulkan_win.so\", \"additional_models\" : [] , \"mode\" : \"interactive\", \"tensor_parallel_shards\" : null, \"pipeline_parallel_stages\" : null, \"gpu_memory_utilization\" : null, \"kv_cache_page_size\" : 16, \"max_num_sequence\" : null, \"max_total_sequence_length\" : null, \"max_single_sequence_length\" : null, \"prefill_chunk_size\" : null, \"sliding_window_size\" : null, \"attention_sink_size\" : null, \"max_history_size\" : null, \"kv_state_kind\" : null, \"speculative_mode\" : \"disable\", \"spec_draft_length\" : 0, \"spec_tree_width\" : 1, \"prefix_cache_mode\" : \"radix\", \"prefix_cache_max_num_recycling_seqs\" : null, \"prefill_mode\" : \"hybrid\", \"verbose\" : true}";
     tokenizer = Tokenizer::FromPath("C:\\Users\\sirki\\teher_workspace\\converted_tiny_llama");
     streamer = TextStreamer(tokenizer);
     Result<EngineCreationOutput> output_res = Engine::Create(
         engine_config, tvm::Device{ static_cast<DLDeviceType>(kDLVulkan), 0 },
         callback_func, {});
     EngineCreationOutput output = output_res.Unwrap(); 
     _generation_config  = output.default_generation_cfg;
     _engine = std::move(output.reloaded_engine);   
 }

 bool LlamaModel::LoadWeights(const std::string& weights_file) {
     return true;
 }

 // Generate texts based on input prompts
 std::string LlamaModel::Process(const std::string prompt) {
         
         //std::vector<std::optional<std::vector<std::vector<std::string>>>> output_logprobs_str(prompts.size());
         size_t num_finished_generations = 0;
         int num_total_generations = 1;
         std::string output_texts;
         bool finished_generation = false;
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
                         std::cout << "finish reason: " << val;
                     }
                 }

             }
             };

         // Override callback in the engine
         GetEngine()->SetRequestStreamCallback(lambda);

         tvm::runtime::String req_id = "012345";
         auto request_data = TextData::TextData(prompt);
         tvm::runtime::Array<Data> inputs;
         inputs.push_back(request_data);

         std::ifstream myFile("C:\\Users\\sirki\\teher_workspace\\converted_tiny_llama\\mlc-chat-config.json");
         std::ostringstream tmp;
         tmp << myFile.rdbuf();
         std::string generation_cfg_json_str = tmp.str();
         auto config_res = json::ParseToJSONObject(generation_cfg_json_str);
         //GenerationConfig def_model_config = GenerationConfig::GetDefaultFromModelConfig(config_res);


         //std::string generation_cfg_json_str = "{\"n\":1,\"temperature\":0.0,\"top_p\":0.0,\"frequency_penalty\":null,\"presence_penalty\":null,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":1024,\"seed\":null,\"stop_strs\":null,\"stop_token_ids\":[2],\"response_format\":null,\"debug_config\":null}";
         //std::string additions_generation_config_str = "{\"n\":1,\"max_tokens\":1024}";
         //auto additions_generation_config = json::ParseToJSONObject(additions_generation_config_str);


         auto combined_config_res = GenerationConfig::FromJSON(config_res, this->_generation_config);
         GenerationConfig g_config = combined_config_res.Unwrap();


         Request req = Request(std::move(req_id), std::move(inputs), g_config);

         GetEngine()->AddRequest(req);

         // Process requests until generation is complete
         while (! finished_generation) {
             GetEngine()->Step();
         }
         return  output_texts;
 }



