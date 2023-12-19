add_library(feature_parser
    feature_parser.cpp
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

target_link_libraries(feature_parser PRIVATE idb)
