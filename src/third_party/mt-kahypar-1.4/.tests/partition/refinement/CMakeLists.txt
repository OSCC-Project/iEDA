target_sources(mt_kahypar_tests PRIVATE
         bipartitioning_gain_policy_test.cc
         refinement_adapter_test.cc
         problem_construction_test.cc
         scheduler_test.cc
#         quotient_graph_test.cc
         gain_policy_test.cc
         label_propagation_refiner_test.cc
         rollback_test.cc
         rebalance_test.cc
         advanced_rebalancer_test.cc
         twoway_fm_refiner_test.cc
         gain_cache_test.cc
         multitry_fm_test.cc
         fm_strategy_test.cc
         flow_construction_test.cc
         )
