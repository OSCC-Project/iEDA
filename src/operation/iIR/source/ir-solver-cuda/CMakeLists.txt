find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

ADD_CUDA_PROJ(ir-solver-cuda)

