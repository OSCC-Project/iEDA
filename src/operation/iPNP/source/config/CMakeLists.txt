add_library(pnp-config PNPConfig.cpp)

target_link_libraries(pnp-config
    PUBLIC
    idb
    idm
)

target_include_directories(pnp-config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_BUILD_TYPE "debug")