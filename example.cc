#include  "include/llama_model.h"

int main() {
    std::string model_path = "C:\\\\Users\\\\sirki\\\\teher_workspace\\\\converted_tiny_llama";
    std::string model_lib_path = "C:\\\\Users\\\\sirki\\\\teher_workspace\\\\tiny_llama_vulkan_win.so";
    LlamaModel model = LlamaModel(model_path, model_lib_path);
    std::string  out = model.Process("what is a meaning of life?");
    std::cout << out << "\n";
    return 0;
}