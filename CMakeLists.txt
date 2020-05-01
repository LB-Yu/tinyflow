CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(tinyflow LANGUAGES C CXX)
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_DIR /usr/local/cuda)

FILE(GLOB CC_SRCS "src/*.cc")
FILE(GLOB CUDA_SRCS "src/*.cu")
FILE(GLOB HEAD_FILES_DIR "src")

INCLUDE_DIRECTORIES(${CUDA_DIR}/include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -Wfatal-errors -Wno-unused -Wno-unused-result")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
SET(ARCH "-gencode arch=compute_30,code=sm_30
          -gencode arch=compute_35,code=sm_35
          -gencode arch=compute_50,code=[sm_50,compute_50]
          -gencode arch=compute_52,code=[sm_52,compute_52]")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --compiler-options '-fPIC' ${ARCH}")

LINK_DIRECTORIES(${CUDA_DIR}/lib64)
CUDA_ADD_LIBRARY(c_runtime_api SHARED ${CC_SRCS} ${CUDA_SRCS})
TARGET_LINK_LIBRARIES(c_runtime_api -lcuda -lcudart -lcublas)

INSTALL(TARGETS c_runtime_api LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/build/lib)