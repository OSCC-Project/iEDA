find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

include_directories(${HOME_UTILITY}/stdBase)
include_directories(${HOME_UTILITY}/stdBase/include)
include_directories(${HOME_UTILITY})
include_directories(${HOME_OPERATION}/iSTA)
include_directories(${HOME_OPERATION}/iSTA/source)
include_directories(${HOME_OPERATION}/iSTA/source/module)
include_directories(${HOME_OPERATION}/iSTA/source/module/include)

include_directories(SYSTEM ${HOME_THIRDPARTY})
include_directories(SYSTEM ${HOME_THIRDPARTY}/yaml-cpp/include)

# set(CMAKE_BUILD_TYPE Release)

link_directories(${CMAKE_BINARY_DIR}/lib)
