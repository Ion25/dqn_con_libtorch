cmake_minimum_required(VERSION 3.10)
project(dqn_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(dqn
  src/main.cpp
  src/dqn_agent.cpp
  src/replay_buffer.cpp
  src/cartpole.cpp
)

target_include_directories(dqn PRIVATE src)
target_link_libraries(dqn "${TORCH_LIBRARIES}")
set_property(TARGET dqn PROPERTY CXX_STANDARD 17)

if(TORCH_CUDA_VERSION)
  message(STATUS "Found Torch with CUDA support: ${TORCH_CUDA_VERSION}")
endif()
