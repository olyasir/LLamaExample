#include  "include/llama_model.h"

std::string get_generation_config( bool use_tiny_llama)
{
    // Create picojson object for generation config
    picojson::object gen_config;
    gen_config["n"] = picojson::value(static_cast<int64_t>(1));
    gen_config["temperature"] = picojson::value(1.0);
    gen_config["top_p"] = picojson::value(1.0);
    gen_config["frequency_penalty"] = picojson::value(0.0);
    gen_config["presence_penalty"] = picojson::value(0.0);
    gen_config["repetition_penalty"] = picojson::value();
    gen_config["logprobs"] = picojson::value(false);
    gen_config["top_logprobs"] = picojson::value(static_cast<double>(0));
    gen_config["logit_bias"] = picojson::value();
    gen_config["max_tokens"] = picojson::value(static_cast<int64_t>(1024));
    gen_config["seed"] = picojson::value();
    picojson::array stop_token_ids;
    if (use_tiny_llama)
    {
        picojson::array stop_strs;
        stop_strs.push_back(picojson::value("</s>"));
        gen_config["stop_strs"] = picojson::value(stop_strs);
        stop_token_ids.push_back(picojson::value(static_cast<int64_t>(2)));
    }else{
        gen_config["stop_strs"] = picojson::value();
        stop_token_ids.push_back(picojson::value(static_cast<int64_t>(128001)));
        stop_token_ids.push_back(picojson::value(static_cast<int64_t>(128008)));
        stop_token_ids.push_back(picojson::value(static_cast<int64_t>(128009)));   
    }   
    gen_config["stop_token_ids"] = picojson::value(stop_token_ids);
    gen_config["response_format"] = picojson::value();
    gen_config["debug_config"] = picojson::value();

    // Convert picojson object to string
     return picojson::value(gen_config).serialize();
}

int main() {
    bool use_tiny_llama = false;
    std::string model_path;
    std::string model_lib_path;

    std::string generation_cfg_json_str = get_generation_config(use_tiny_llama);
    if(use_tiny_llama){
        //generation_cfg_json_str = "{\"n\":1,\"temperature\":1.0,\"top_p\":1.0,\"frequency_penalty\":0,\"presence_penalty\":0,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":1024,\"seed\":null,\"stop_strs\":[\"</s>\"],\"stop_token_ids\":[2],\"response_format\":null,\"debug_config\":null}";
        model_path = "/home/ubuntu/compiled_models/TinyLlama-1.1B-Chat-v1.0";
        model_lib_path = "/home/ubuntu/compiled_models/libs/TinyLlama-1.1B-Chat-v1.0-vulkan.so";

    }else{
        //generation_cfg_json_str = "{\"n\":1,\"temperature\":0.6,\"top_p\":0.9,\"frequency_penalty\":0,\"presence_penalty\":0,\"repetition_penalty\":null,\"logprobs\":false,\"top_logprobs\":0,\"logit_bias\":null,\"max_tokens\":1024,\"seed\":null,\"stop_strs\":null,\"stop_token_ids\":[128001, 128008, 128009],\"response_format\":null,\"debug_config\":null}"; 
        model_path = "/home/ubuntu/compiled_models/Meta-Llama-3.1-8B-Instruct";
        model_lib_path = "/home/ubuntu/compiled_models/libs/Meta-Llama-3.1-8B-Instruct-vulkan.so";
    }
    
    LlamaModel model = LlamaModel(model_path, model_lib_path, generation_cfg_json_str);

    while (true)
    {

        std::string prompt ;
        std:: cout << "Enter prompt or 'exit' to finish.\n >>"; 
        getline(std::cin, prompt);
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