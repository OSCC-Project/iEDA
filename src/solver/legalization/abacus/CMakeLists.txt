add_library(method_abacus
    AbacusCluster.cc
    Abacus.cc
)

target_link_libraries(method_abacus
    PRIVATE
    ipl-module-logger
    ipl-bridge
    ipl-configurator

)

target_include_directories(method_abacus PUBLIC ${ABACUS_METHOD})