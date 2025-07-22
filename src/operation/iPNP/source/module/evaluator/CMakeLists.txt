add_library(evaluator CongestionEval.cpp IREval.cpp FastPlacer.cpp DRCEval.cpp)

target_link_libraries(evaluator PUBLIC power ista-engine eval_density_api eval_congestion_api eval_db ipl-api
                                       data_manager)
