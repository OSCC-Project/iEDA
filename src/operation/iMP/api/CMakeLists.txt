add_library(imp_api
    ${IMP_API}/MPAPI.cc
    
)

target_link_libraries(imp_api
    PRIVATE
        imp_source
        idm
        imp-parser
        imp-structure
)

target_include_directories(imp_api
    PUBLIC
    ${IMP_API}
    ${HOME_IMP}/source
    ${HOME_IMP}/source/data_manager/basic
    ${HOME_IMP}/source/data_manager/parser
    ${HOME_IMP}/source/utility/structure
)
