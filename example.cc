#include  "include/llama_model.h"

int main() {
    LlamaModel model;
    std::string  out = model.Process("what is a meaning of life?");
    std::cout << out << "\n";
    return 0;
}