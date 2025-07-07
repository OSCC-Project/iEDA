cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)

aux_source_directory(./ SRC)

add_library(notification STATIC ${SRC})
target_link_libraries(notification PRIVATE curl)

target_include_directories(notification
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CURL_INCLUDE_DIRS}
        ${HOME_THIRDPARTY}/json
)
