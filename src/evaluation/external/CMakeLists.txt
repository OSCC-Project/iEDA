## include path
include(${EVAL_EXTERNAL}/idb.cmake)
include(${EVAL_EXTERNAL}/iutility.cmake)
include(${EVAL_EXTERNAL}/third_party.cmake)
include(${EVAL_EXTERNAL}/ista.cmake)
include(${EVAL_EXTERNAL}/iplf.cmake)

add_library(eval_source_external_libs INTERFACE)
target_link_libraries(eval_source_external_libs
    INTERFACE
        eval_idb_libs
        eval_iutility_libs
        eval_third_party_libs
        eval_ista_libs
        eval_iplf_libs
)
