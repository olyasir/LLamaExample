cmake_minimum_required(VERSION 3.25)
project(llama_example)


set(CMAKE_BUILD_TYPE Debug)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "-fPIC -std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0 ${CMAKE_C_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
endif()

# Add mlc-llm subdirectory
add_subdirectory(mlc-llm)



set(MLC_LLM_HOME ${PROJECT_SOURCE_DIR}/mlc-llm)

set(MLC_LLM_TVM_HOME "C:\\Users\\sirki\\teher_workspace\\tvm")
set(MLC_LLM_TVM_INCLUDE_DIRS ${MLC_LLM_TVM_HOME}/include
                             ${MLC_LLM_TVM_HOME}/3rdparty/dlpack/include
                             ${MLC_LLM_TVM_HOME}/3rdparty/dmlc-core/include
                             ${MLC_LLM_TVM_HOME}/3rdparty/picojson)

set(MLC_LLM_TOKENIZERS_INCLUDES ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include)


# Add the executable
add_executable(llama_model llama_model.cc)


#link_directories([AFTER|BEFORE] directory1 [directory2 ...])
# Link libraries
target_link_libraries(llama_model PUBLIC tvm_runtime)
target_link_libraries(llama_model PRIVATE tokenizers_cpp)
target_link_libraries(llama_model PRIVATE mlc_llm)
target_link_libraries(llama_model PRIVATE mlc_llm_module)



target_include_directories(llama_model PRIVATE ${MLC_LLM_HOME}
                                                 ${MLC_LLM_TVM_INCLUDE_DIRS}
                                                 ${MLC_LLM_TOKENIZERS_INCLUDES})


# Include directories
target_include_directories(llama_model PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/mlc-llm/include
    ${CMAKE_CURRENT_BINARY_DIR}/mlc-llm
     ${CMAKE_CURRENT_BINARY_DIR}
)