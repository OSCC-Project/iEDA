add_library(feature_parser
    feature_parser.cpp
    feature_parser_summary.cpp
)

target_include_directories(feature_parser 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}

        ##json
        ${HOME_THIRDPARTY}/json
        ${HOME_UTILITY}/json

        ##auxiliary
        ${HOME_UTILITY}
        ${HOME_UTILITY}/json
)

target_link_libraries(feature_parser 
    PRIVATE 
        idb
        flow_config
        idm
        ipl-source
        ipl-api
        icts_api
        ista-engine
        idrc_api
        eval_api
        ieda_report_evaluator
)
