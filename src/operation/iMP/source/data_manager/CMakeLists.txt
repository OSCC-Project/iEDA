add_library(imp-datamanager
    ${IMP_DATAMANAGER}/DataManager.cc
)

target_link_libraries(imp-datamanager
    PRIVATE
    imp-wrapper
    imp-logger
)
target_include_directories(imp-datamanager
    PUBLIC
        ${IMP_DATAMANAGER}
        ${IMP_DATAMANAGER}/basic
)
set(IMP_WRAPPER ${IMP_DATAMANAGER}/wrapper)
add_subdirectory(${IMP_WRAPPER})