set(iPL_EXTERNAL_API ${iPL_API}/external_api)
set(iPL_REPORTER ${iPL_API}/report)

add_subdirectory(${iPL_EXTERNAL_API})
add_subdirectory(${iPL_REPORTER})

add_library(ipl-api
    ${iPL_API}/PLAPI.cc
)

target_link_libraries(ipl-api
    PRIVATE
        ipl-source
    PUBLIC
        ipl-reporter
        ipl-external_api
)

target_include_directories(ipl-api
    PUBLIC
        ${iPL_API}
)