add_library(evaluator
    CongestionEval.cpp
    IREval.cpp
)

target_link_libraries(evaluator PUBLIC power ista-engine idm eval_density_api data_manager)
 