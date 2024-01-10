add_library(imp-operator BlkClustering.cc)
target_link_libraries(imp-operator
    PRIVATE
    imp-partition
    imp-basic
    imp-logger
)
target_include_directories(imp-operator
    PUBLIC
    ${IMP_OPERATOR}
)