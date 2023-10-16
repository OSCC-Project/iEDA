add_library(ipl-external_api
    ${iPL_EXTERNAL_API}/ExternalAPI.cc
)

target_link_libraries(ipl-external_api
    PUBLIC
        ipl-api_external_libs
)

target_include_directories(ipl-external_api
    PUBLIC
        ${iPL_EXTERNAL_API}
)