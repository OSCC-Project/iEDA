add_library(ito_api
    ToApi.cpp
)

target_link_libraries(ito_api
    PRIVATE
        ito_source
        # ito_api_external_libs
        ito_data_manager
        ito_config
        idm
        ista-engine
        ito_timing_engine
)

target_include_directories(ito_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
