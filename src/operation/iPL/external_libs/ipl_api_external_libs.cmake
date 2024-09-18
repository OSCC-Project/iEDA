add_library(ipl-api_external_libs INTERFACE)

find_package(OpenMP REQUIRED)
target_link_libraries(ipl-api_external_libs
    INTERFACE
        eval_pro_congestion_api
        eval_pro_wirelength_api
        eval_pro_timing_api
        eval_pro_density_api
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
        ${HOME_EVALUATION_PRO}/api
        ${HOME_EVALUATION_PRO}/database

)
