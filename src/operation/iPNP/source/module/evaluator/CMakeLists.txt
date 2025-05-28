add_library(evaluator CongestionEval.cpp IREval.cpp FastPlacer.cpp)

target_link_libraries(evaluator PUBLIC power ista-engine eval_density_api
                                       data_manager idm)
