cmake_minimum_required(VERSION 3.25)
project(llama_example LANGUAGES CXX C)


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


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "")
if(APPLE)
  set(USE_METAL ON CACHE BOOL "Use the Metal API for TVM runtime")
else()
  set(USE_VULKAN ON CACHE BOOL "Use the Vulkan API for TVM runtime")
endif()


# Add mlc-llm subdirectory
add_subdirectory(mlc-llm)


message(PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR})

set(MLC_LLM_HOME ${PROJECT_SOURCE_DIR}/mlc-llm)


set(TVM_HOME ${MLC_LLM_HOME}/3rdparty/tvm)
message(TVM_HOME=${TVM_HOME})

set(TVM_INCLUDES ${TVM_HOME}/include
                 ${TVM_HOME}/3rdparty/dlpack/include
                 ${TVM_HOME}/3rdparty/dmlc-core/include
                 ${TVM_HOME}/3rdparty/picojson)
set(TOKENIZERS_INCLUDES ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include)




#set(MLC_LLM_TVM_HOME "C:\\Users\\sirki\\teher_workspace\\tvm")
#set(MLC_LLM_TVM_INCLUDE_DIRS ${MLC_LLM_TVM_HOME}/include
#                             ${MLC_LLM_TVM_HOME}/3rdparty/dlpack/include
#                             ${MLC_LLM_TVM_HOME}/3rdparty/dmlc-core/include
#                             ${MLC_LLM_TVM_HOME}/3rdparty/picojson)

#set(MLC_LLM_TOKENIZERS_INCLUDES ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include)


set(LIB_NAME llama_model_base)
set(LIB_SOURCES include/llama_model.h
                src/llama_model.cc)

add_library(${LIB_NAME} STATIC ${LIB_SOURCES})
add_executable(cli example.cc)
add_executable(cli_tool tool_example.cc)

#link_directories([AFTER|BEFORE] directory1 [directory2 ...])
# Link libraries
#target_link_libraries(llama_model PUBLIC tvm_runtime)
#target_link_libraries(llama_model PRIVATE tokenizers_cpp)
#target_link_libraries(llama_model PRIVATE mlc_llm)
#target_link_libraries(llama_model PRIVATE mlc_llm_module)



target_include_directories(${LIB_NAME} PUBLIC ${MLC_LLM_HOME}
                                                 ${TVM_INCLUDES}
                                                 ${TOKENIZERS_INCLUDES})

target_include_directories(cli PRIVATE ${PROJECT_SOURCE_DIR} 
                                        ${MLC_LLM_HOME}
                                        ${TVM_INCLUDES}
                                        ${TOKENIZERS_INCLUDES})

target_include_directories(cli_tool PRIVATE ${PROJECT_SOURCE_DIR} 
                                        ${MLC_LLM_HOME}
                                        ${TVM_INCLUDES}
                                        ${TOKENIZERS_INCLUDES})


# Include directories
target_link_libraries(${LIB_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,mlc_llm_static,tvm_runtime,tokenizers_cpp>")
target_link_libraries(cli PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
target_link_libraries(cli_tool PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")

