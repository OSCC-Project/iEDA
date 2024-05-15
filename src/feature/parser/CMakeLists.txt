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
        ${HOME_DATABASE}/interaction/Summary
)

target_link_libraries(feature_parser 
    PRIVATE 
        idb
        flow_config
        idm
        irt_interface
        ipl-source
        ipl-api
        icts_api
        ista-engine
        idrc_pro_api
        eval_api
        ieda_report_evaluator
        feature_db
)
