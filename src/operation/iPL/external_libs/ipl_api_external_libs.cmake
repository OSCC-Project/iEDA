add_library(ipl-api_external_libs INTERFACE)

find_package(OpenMP REQUIRED)
target_link_libraries(ipl-api_external_libs
    INTERFACE
        eval_congestion_api
        eval_wirelength_api
        eval_timing_api
        eval_density_api
        tool_manager
        idm
        OpenMP::OpenMP_CXX
        report_table
        feature_db
)

target_include_directories(ipl-api_external_libs
    INTERFACE
        ${HOME_PLATFORM}
        ${HOME_PLATFORM}/tool_manager
        ${HOME_PLATFORM}/data_manager
        ${HOME_EVALUATION}/api
        ${HOME_EVALUATION}/database

)
