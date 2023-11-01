project(TestCPP)

set(CMAKE_CXX_STANDARD 11)  # 设置 C++ 标准

add_executable(test_cpp test.cpp)  # 添加编译的源文件

# 如果需要链接其他库，添加链接选项
# target_link_libraries(test_cpp your_additional_libraries)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
