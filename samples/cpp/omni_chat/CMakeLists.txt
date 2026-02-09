cmake_minimum_required(VERSION 3.15)
project(omni_chat)

find_package(OpenVINOGenAI REQUIRED)

add_executable(omni_chat omni_chat.cpp)
target_link_libraries(omni_chat PRIVATE openvino::genai)
set_target_properties(omni_chat PROPERTIES CXX_STANDARD 17)

install(TARGETS omni_chat
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)
