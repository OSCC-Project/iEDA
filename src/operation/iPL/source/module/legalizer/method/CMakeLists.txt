include_directories(${LG_METHOD})
set(ABACUS_METHOD ${LG_METHOD}/abacus)

add_subdirectory(${ABACUS_METHOD})

add_library(legalizer_method INTERFACE)

target_link_libraries(legalizer_method 
    INTERFACE
        legalizer_method_abacus
)

target_include_directories(legalizer_method INTERFACE ${LG_METHOD})