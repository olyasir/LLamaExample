#include "llama_model.h"
#include <iostream>
#include <vector>
#include <cpp/serve/model.h>
#include <fstream>
#include <picojson.h>
#include <tokenizers_cpp.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cpp/serve/config.h>

using tokenizers::Tokenizer;
using namespace mlc::llm::serve ;

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

int main() {
 
  // Read blob from file.
  auto blob = LoadBytesFromFile("/home/ubuntu/olya/mlc-llm/dist/Meta-Llama-3.1-8B-Instruct/tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);
  std::string prompt = "What is the capital of Canada?";
  // call Encode to turn prompt into token ids
  std::vector<int> input_tokens = tok->Encode(prompt);
  

  // Read and parse the JSON file
  //const String& model_str  = 
  tvm::runtime::String model_str = "/home/ubuntu/olya/mlc-llm/dist/Meta-Llama-3.1-8B-Instruct";
  picojson::object model_config = mlc::llm::serve::Model::LoadModelConfig(model_str).Unwrap();
     
  mlc::llm::serve::Model model = mlc::llm::serve::Model::Create(
    "/home/ubuntu/olya/mlc-llm/dist/libs/Meta-Llama-3.1-8B-Instruct-vulkan.so",
    "/home/ubuntu/olya/mlc-llm/dist/Meta-Llama-3.1-8B-Instruct",
    model_config,
    DLDevice{kDLCPU, 7},
    {},
    1,
    1,
    false
  );
std::cout << "Model created" << std::endl;
model->LoadParams();

std::cout << "Model loaded params" << std::endl;
model->SetPrefillChunkSize(2048);
model->SetMaxNumSequence(131072);


model->CreateKVCache(16, 4,
                           131072,
                           2048, 0);

std::cout << "Created kv cache" << std::endl;

// n->model_workspaces_.push_back(
 ObjectRef embeddings =  model->AllocEmbeddingTensor();
 ObjectRef hidden_states  =  model->AllocHiddenStatesTensor();

model->TokenEmbed({IntTuple(input_tokens.begin(), input_tokens.end())}, &embeddings);

std::cout << "Model Finished Embeddings" << std::endl;
std::vector<int> prefill_lengths;
int num_rsentries = 1;
prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
prefill_lengths[0] = input_tokens.size();
model->AddNewSequence(0);
NDArray logits = model->BatchPrefill(embeddings, {0}, prefill_lengths);
std::cout << "Model Finished Prefill" << std::endl;

int max_num_tokens = 128256;
Optional<EventTraceRecorder> opt0 = Optional<EventTraceRecorder>();
LogitProcessor logit_processor = model->CreateLogitProcessor(2048, opt0);
Sampler sampler = model->CreateSampler( max_num_tokens, 1, {});
std::cout << "Created logits processor" << std::endl;

//     // - Update logits.
  logits = logits.CreateView({num_rsentries, logits->shape[2]}, logits->dtype);
  GenerationConfig generation_cfg = GenerationConfig::GetDefaultFromModelConfig(model_config);

  Array<RequestModelState>mstates;
  mstates.push_back(RequestModelState({}, 0, 0, {}, {}));
  
  logit_processor->InplaceUpdateLogits(logits, {generation_cfg}, mstates, {});

    // - Compute probability distributions.
    NDArray probs_on_device =
        logit_processor->ComputeProbsFromLogits(logits, {generation_cfg}, {});

    // - Sample tokens.
    // Fill range [0, num_rsentries) into `sample_indices`.
    std::vector<int> sample_indices(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    NDArray renormalized_probs = sampler->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, {}, {generation_cfg});
    std::vector<SampleResult> sample_results = sampler->BatchSampleTokensWithProbAfterTopP(
        renormalized_probs, sample_indices, {}, {generation_cfg}, {});
   

//     // - Update the committed tokens of states.
//     for (int i = 0; i < num_rsentries; ++i) {
//       mstates[i]->CommitToken(sample_results[i]);
//       // Metrics update
//       // live update the output metrics
//       running_rsentries[i]->rstate->metrics.completion_tokens += 1;
//     }

// // call Decode to turn ids into string
//   std::string decoded_prompt = tok->Decode(ids);
  

  return 0;
}