file(GLOB SALT_SRCS *.cpp */*.cpp base/flute/*.c)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
add_library(salt STATIC ${SALT_SRCS})
