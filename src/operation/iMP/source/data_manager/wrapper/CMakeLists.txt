add_library(imp-wrapper
            IDBWrapper.cc)

target_link_libraries(imp-wrapper
    PUBLIC
    idm
)

target_include_directories(imp-wrapper 
    PUBLIC 
    ${IMP_WRAPPER}
    ${IMP_DATAMANAGER}/basic)