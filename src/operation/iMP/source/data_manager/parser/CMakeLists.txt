add_library(imp-parser
            IDBParserEngine.cc)

target_link_libraries(imp-parser
    PUBLIC
    idm
    imp-basic
)

target_include_directories(imp-parser
    PUBLIC 
    ${IMP_PARSER}
    ${IMP_BASIC}
    )