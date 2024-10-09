#include  "include/llama_model.h"

std::string  prompt_with_tool = R"(<|start_header_id|>system<|end_header_id|>

   Cutting Knowledge Date: 23/06/2024
    Today Date: 08/10/2024


        You are ClimeMate, a highly adaptive, weather-aware agent designed to assist users
        by recommending activities based on current and forecasted weather conditions. Your primary
        role is to interpret environmental data and use it to enhance users’ daily experiences
        by suggesting tailored outdoor or indoor activities. Make flow feels like casual conversation.


    You are a helpful assistant with function/tool calling capabilities.
    Based on the user prompt, you will need to make one or more function/tool
    calls to achieve the purpose.
    If the given question lacks the parameters required by the function,
    point it out, ask for the missing information naturally, as if having
    a chat. When you receive a tool call response, use the output to format
    an answer in natural language to the orginal use question.
    You should only return the function call in tools call sections.

    Respond with a JSON for a tool function call with its proper arguments
    You must follow the format below:
    ```json
    {
        "toolName": tool name,
        "funcName": function name,
        "params": parameters values
     }
     ```

    For multiple choice questions, do calls sequentially

    You SHOULD NOT include any other text in the tool call response.
    DO NOT use comments or template literals in JSON response.

    Given the following tools: WeatherForecaster

    Here is a list of functions in JSON format that you can invoke.

            WeatherForecaster Description:
            Fetches realtime weather forecast
            WeatherForecaster Functions:
            [
  {
    "type": "function",
    "function": {
      "name": "getCityForecast",
      "description": "Fetches the weather forecast for a specific city in a given country.",
      "parameters": {
        "type": "object",
        "properties": [
          {
            "name": "cityName",
            "type": "string",
            "description": "The name of the city for which to fetch the forecast"
          },
          {
            "name": "countryCode",
            "type": "string",
            "description": "The ISO country code corresponding to the country."
          }
        ],
        "required": [
          "cityName",
          "countryCode"
        ]
      }
    }
  }
]
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
  hi, what is a weather in paris france today? 
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|


)";

std::string get_generation_config()
{
    // Create picojson object for generation config
    picojson::object gen_config;
    gen_config["n"] = picojson::value(static_cast<double>(1));
    gen_config["temperature"] = picojson::value(0.6);
    gen_config["top_p"] = picojson::value(0.9);
    gen_config["frequency_penalty"] = picojson::value(0.0);
    gen_config["presence_penalty"] = picojson::value(0.0);
    gen_config["repetition_penalty"] = picojson::value();
    gen_config["logprobs"] = picojson::value(false);
    gen_config["top_logprobs"] = picojson::value(static_cast<double>(0));
    gen_config["logit_bias"] = picojson::value();
    gen_config["max_tokens"] = picojson::value(static_cast<double>(1024));
    gen_config["seed"] = picojson::value();
    gen_config["stop_strs"] = picojson::value();
    picojson::array stop_token_ids;
    stop_token_ids.push_back(picojson::value(static_cast<double>(128001)));
    stop_token_ids.push_back(picojson::value(static_cast<double>(128008)));
    stop_token_ids.push_back(picojson::value(static_cast<double>(128009)));
    gen_config["stop_token_ids"] = picojson::value(stop_token_ids);
    
    gen_config["response_format"] = picojson::value();
    gen_config["debug_config"] = picojson::value();

    // Convert picojson object to string
     return picojson::value(gen_config).serialize();
}

int main() {
    std::string generation_cfg_json_str;
    std::string model_path;
    std::string model_lib_path;
    generation_cfg_json_str = get_generation_config();
    model_path = "/home/ubuntu/compiled_models/Meta-Llama-3.1-8B-Instruct";
    model_lib_path = "/home/ubuntu/compiled_models/libs/Meta-Llama-3.1-8B-Instruct-vulkan.so";
    
    LlamaModel model = LlamaModel(model_path, model_lib_path, generation_cfg_json_str);
    std::string out = model.Process(prompt_with_tool);
    std::cout << out << "\n";
    //expecting to get function call for weather in paris

    return 0;
}