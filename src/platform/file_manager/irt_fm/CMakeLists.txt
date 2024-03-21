
# 此cmake有问题
# 此cmake有问题
# 此cmake有问题
# 此cmake有问题
# 此cmake有问题

# cmake_minimum_required(VERSION 3.10)

# add_library(file_manager_routing
#     file_rt.cpp
# )

# find_package(Boost COMPONENTS system serialization REQUIRED)

# target_link_libraries(file_manager_routing 
#     PUBLIC 
#         file_manager_base  
#         不能这样引cmake
#         irt_source
#         Boost::boost 
#         Boost::serialization
# )

# target_include_directories(file_manager_routing
#     PUBLIC
#         ${CMAKE_CURRENT_SOURCE_DIR}
# )


# option(TEST_RT_SERIALIZE "If ON, test db." OFF)

# if(TEST_RT_SERIALIZE)
#     find_package(GTest REQUIRED)
#     add_executable(test_rt_serialize test.cpp)

#     target_include_directories(test_rt_serialize
#         PRIVATE
#             ${GTEST_INCLUDE_DIRS}
#     )

#     target_link_libraries(test_rt_serialize 
#         file_manager_routing 
#         ${GTEST_LIBRARIES} 
#         ${GTEST_MAIN_LIBRARIES} 
#     )
# endif()