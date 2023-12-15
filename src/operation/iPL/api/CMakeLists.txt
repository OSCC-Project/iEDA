set(iPL_EXTERNAL_API ${iPL_API}/external_api)
add_subdirectory(${iPL_EXTERNAL_API})

add_library(ipl-api
    ${iPL_API}/PLAPI.cc
    ${iPL_API}/report/PLReport.cc
)

target_link_libraries(ipl-api
    PRIVATE
        ipl-source
    PUBLIC
        ipl-external_api
)

target_include_directories(ipl-api
    PUBLIC
        ${iPL_API}
)