add_library(legalizer_method_abacus
    AbacusCluster.cc
    Abacus.cc
)

target_link_libraries(legalizer_method_abacus
    PRIVATE
    ipl-module-logger
    ipl-bridge
    ipl-configurator

)

target_include_directories(legalizer_method_abacus PUBLIC ${ABACUS_METHOD})