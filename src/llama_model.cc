#include "../include/llama_model.h"



using namespace mlc::llm;
using namespace mlc::llm::serve;

loglevel_e loglevel = logDEBUG;
class LlamaEngineImpl;

void  callback_func(Array<RequestStreamOutput> out){}

std::string  get_engine_config(String model_path, String model_lib)
{
    picojson::object config;

    config["model"] = picojson::value(model_path);
    config["model_lib"] = picojson::value(model_lib);
    config["additional_models"] = picojson::value(picojson::array());
    config["mode"] = picojson::value("interactive");
    config["tensor_parallel_shards"] = picojson::value();
    config["pipeline_parallel_stages"] = picojson::value();
    config["gpu_memory_utilization"] = picojson::value();
    config["kv_cache_page_size"] = picojson::value(static_cast<double>(16));
    config["max_num_sequence"] = picojson::value();
    config["max_total_sequence_length"] = picojson::value();
    config["max_single_sequence_length"] = picojson::value();
    config["prefill_chunk_size"] = picojson::value();
    config["sliding_window_size"] = picojson::value();
    config["attention_sink_size"] = picojson::value();
    config["max_history_size"] = picojson::value();
    config["kv_state_kind"] = picojson::value();
    config["speculative_mode"] = picojson::value("disable");
    config["spec_draft_length"] = picojson::value(static_cast<double>(0));
    config["spec_tree_width"] = picojson::value(static_cast<double>(1));
    config["prefix_cache_mode"] = picojson::value("radix");
    config["prefix_cache_max_num_recycling_seqs"] = picojson::value();
    config["prefill_mode"] = picojson::value("hybrid");
    config["verbose"] = picojson::value(false);

    return picojson::value(config).serialize();
}


 LlamaModel::LlamaModel(String model_path, String model_lib, 
                        std::string generation_config_str, tvm::Device device ) {
     log(logDEBUG) << "Model Path "    << model_path << " \n";
     log(logDEBUG) << "model_lib "    << model_lib << " \n";
     std::string engine_config = get_engine_config(model_path, model_lib);
     tokenizer = Tokenizer::FromPath(model_path);
     log(logDEBUG) << "Loaded Tokenizer..\n";
     
     log(logDEBUG) << "Started Engine Creation..\n";
     Result<LLamaEngineCreationOutput> output_res = LlamaEngine::Create(
         engine_config, device,
         callback_func, {});
     log(logDEBUG) << "Finished Engine creation..\n";
     LLamaEngineCreationOutput output = output_res.Unwrap(); 
     GenerationConfig default_generation_config  = output.default_generation_cfg;
     auto additions_generation_config = json::ParseToJSONObject(generation_config_str);
     auto combined_config_res = GenerationConfig::FromJSON(additions_generation_config, default_generation_config);
    _generation_config = combined_config_res.Unwrap();
     _engine = std::move(output.reloaded_engine);   
     _model_path = model_path;
     _device = device;
     _loaded_weight_num = 0;
    // _model_metadata  = output.model_metadata;

     std::filesystem::path fs_model_path = _model_path;

     std::string metadata_path = (fs_model_path / "ndarray-cache.json").string();

      std::string ndarray_cache_json = get_file_contents(metadata_path);
      ndarray_cache_metadata_ = NDArrayCacheMetadata::LoadFromStr(
      ndarray_cache_json,
      ""
    );
    _required_weight_num = ndarray_cache_metadata_.records.size();
     log(logDEBUG) << "Finished model configuration initialization\n";
     _initialized = false;
 }



 void UpdateNDArrayCache(
                          const NDArrayCacheMetadata& ndarray_cache_metadata,
                          const std::string& filename,
                          std::vector<uint8_t> weight_data,
                          bool use_presharded_weights,
                           Device device) {
    using tvm::runtime::ShapeTuple;
    using FileRecord = NDArrayCacheMetadata::FileRecord;
    using ParamRecord = FileRecord::ParamRecord;

    const size_t filename_chars = std::string("params_shard_").size();
    size_t stop_idx = filename.find_last_of('.');
    std::string remaining = filename.substr(filename_chars, stop_idx - filename_chars);
    size_t file_record_idx = std::stoi(remaining);
    const FileRecord& file_record = ndarray_cache_metadata.records[file_record_idx];
    size_t total_param_records = file_record.records.size();
    Array<NDArray> params;
    const auto& params_shard_file = weight_data;
    const std::string raw_data_buffer { params_shard_file.begin(), params_shard_file.end() };
    Optional<NDArray> staging_buffer;

    CHECK(!use_presharded_weights) << "Use of pre-sharded weights requires more than one GPU";
    std::cerr << filename << " has these many parameter records: " << total_param_records << '\n';

    params.reserve(total_param_records);

    for (size_t i = 0; i < total_param_records; ++i) {
      const ParamRecord& param_record = file_record.records[i];

      params.push_back(param_record.Load(device,
                                         &raw_data_buffer,
                                         &staging_buffer));
    }

    const PackedFunc* fload_cache_update = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.update");
    ICHECK(fload_cache_update) << "TVM runtime cannot find vm.builtin.ndarray_cache.update";

    /* Update the global cache with all the various parameters */
    for (size_t i = 0; i < params.size(); ++i)
      (*fload_cache_update)(file_record.records[i].name, params[i], true);
  }



 bool LlamaModel::LoadWeights(const std::string& weights_file) {
    //TODO check if file was not already loaded
    log(logDEBUG) << "Adding weights" << weights_file <<"\n";
    std::string full_path = std::string("").append(_model_path).append("/"+weights_file);
    std::vector<uint8_t> weight_data = get_bytes_from_file(full_path);

    UpdateNDArrayCache(ndarray_cache_metadata_, weights_file, weight_data, false, _device);

    //load file to cache
    _loaded_weight_num++;
    if (_loaded_weight_num == _required_weight_num)
    {
        log(logDEBUG) << "Loaded all weights starting to load parameters \n";
         _initialized = true;
         LoadParams();
        //load params from cache
    }

     return true;
 }



void   LlamaModel::LoadParams() {
    GetEngine()->LoadParams();
  }

std::string LlamaModel::Metrics(){
    return GetEngine()->JSONMetrics();
}

 // Generate texts based on input prompts
 std::string LlamaModel::Process(const std::string prompt) {
        if ( !_initialized){
            return "Cannot generate response. Please load model weights\n";
        }
        log(logDEBUG) << "Start Processing prompt"<<prompt<<"\n";
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



