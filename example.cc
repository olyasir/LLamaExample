#include  "include/llama_model.h"

int main() {
    bool use_tiny_llama = true;
    std::string generation_cfg_json_str;
    std::string model_path;
    std::string model_lib_path;


    if(use_tiny_llama){
        generation_cfg_json_str = "{\"n\":1,\"temperature\":1.0,\"top_p\":1.0,\"frequency_penalty\":0,\"presence_penalty\":0,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":1024,\"seed\":null,\"stop_strs\":[\"</s>\"],\"stop_token_ids\":[2],\"response_format\":null,\"debug_config\":null}";
        model_path = "/home/ubuntu/compiled_models/TinyLlama-1.1B-Chat-v1.0";
        model_lib_path = "/home/ubuntu/compiled_models/libs/TinyLlama-1.1B-Chat-v1.0-vulkan.so";

    }else{
        generation_cfg_json_str = "{\"n\":1,\"temperature\":0.6,\"top_p\":0.9,\"frequency_penalty\":0,\"presence_penalty\":0,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":1024,\"seed\":null,\"stop_strs\":null,\"stop_token_ids\":[128001, 128008, 128009],\"response_format\":null,\"debug_config\":null}"; 
        model_path = "/home/ubuntu/compiled_models/Meta-Llama-3.1-8B-Instruct";
        model_lib_path = "/home/ubuntu/compiled_models/libs/Meta-Llama-3.1-8B-Instruct-vulkan.so";
    }
    
    LlamaModel model = LlamaModel(model_path, model_lib_path, generation_cfg_json_str);

    while (true)
    {

        std::string prompt ;
        std:: cout << "Enter prompt or 'exit' to finish.\n >>"; 
        getline(std::cin, prompt);
        std::cout<< "The prompt is : "<<prompt<<"\n";
        if (prompt == "exit")
            break;

        std::string tiny_llama_input = R"(Message: <|system|>
        You are a helpful chatbot.</s><|user|>
        )"+ prompt + R"(</s><|assistant|>

        )";
            
        std::string input = R"(<|start_header_id|>system<|end_header_id|>

        You are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        )" +prompt+R"(<|eot_id|><|start_header_id|>assistant<|end_header_id|>


        )";

        std::string out = model.Process(use_tiny_llama ? tiny_llama_input : input);
        std::cout << out << "\n";
    }
    return 0;
}