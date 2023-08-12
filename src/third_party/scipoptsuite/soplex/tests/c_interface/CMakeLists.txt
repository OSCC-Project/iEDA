project(soplex_testing LANGUAGES C)

add_executable(soplex_c_testing "main.c")
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/../../src/")
target_link_libraries(soplex_c_testing libsoplexshared)
