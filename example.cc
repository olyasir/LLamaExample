#include  "include/llama_model.h"

int main() {
    std::string model_path = "/home/ubuntu/compiled_models/Meta-Llama-3.1-8B-Instruct";
    std::string model_lib_path = "/home/ubuntu/compiled_models/libs/Meta-Llama-3.1-8B-Instruct-vulkan.so";
    LlamaModel model = LlamaModel(model_path, model_lib_path);
    
    // Using a multiline string for the input prompt
    std::string input = R"(<|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>


)";

    std::string out = model.Process(input);
    std::cout << out << "\n";
    return 0;
}