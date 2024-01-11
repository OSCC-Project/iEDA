add_library(ipl-reporter
    ${iPL_REPORTER}/PLReporter.cc
)

target_link_libraries(ipl-reporter
    PRIVATE
        ipl-source
    PUBLIC
        ipl-external_api
)

target_include_directories(ipl-reporter
    PUBLIC
        ${iPL_REPORTER}
)