SET(CMAKE_BUILD_TYPE "Release")

include_directories(${HOME_DATABASE}/manager/parser)

add_compile_options(-w)
add_subdirectory(aocv)
add_subdirectory(feature)
add_subdirectory(gdsii)
add_subdirectory(verilog)
add_subdirectory(liberty)
add_subdirectory(vcd)
