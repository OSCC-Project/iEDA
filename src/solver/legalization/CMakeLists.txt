set(LG_METHOD ${HOME_SOLVER}/legalization)
include_directories(${LG_METHOD})

set(PL_SOURCE ${HOME_OPERATION}/iPL/source)
set(PL_LGDATASOURCE ${HOME_OPERATION}/iPL/source/module/legalizer)
message("Test: " ${HOME_OPERATION})


set(ABACUS_METHOD ${LG_METHOD}/abacus)
add_subdirectory(${ABACUS_METHOD})

add_library(solver_legalization INTERFACE)

target_link_libraries(solver_legalization 
    INTERFACE
        method_abacus
)

target_include_directories(method_abacus INTERFACE ${PL_SOURCE})
target_include_directories(method_abacus INTERFACE ${PL_LGDATASOURCE})
target_include_directories(method_abacus INTERFACE ${PL_LGDATASOURCE}/config)
target_include_directories(method_abacus INTERFACE ${PL_LGDATASOURCE}/database)

target_include_directories(method_abacus INTERFACE ${LG_METHOD})

message("-- Solver-Legalization: ENABLE")