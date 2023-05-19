include_directories(${iPL_MP})

add_library(ipl-module-macro_placer
    ${iPL_MP}/MacroPlacer.cc
    ${iPL_MP}/MPDB.cc
    ${iPL_MP}/wrapper/IPLDBWrapper.cc
    ${iPL_MP}/partition/MPPartition.cc
    ${iPL_MP}/partition/HierPartition.cc
    ${iPL_MP}/partition/Module.cc
    ${iPL_MP}/analytical_method/Solver.cc
    ${iPL_MP}/analytical_method/Problem.cc
    ${iPL_MP}/analytical_method/MProblem.cc
    ${iPL_MP}/simulate_anneal/BStarTree.cc
    ${iPL_MP}/simulate_anneal/SequencePair.cc
    ${iPL_MP}/simulate_anneal/MPEvaluation.cc)

target_include_directories(ipl-module-macro_placer
    PUBLIC
    ${iPL_MP}
)

target_link_libraries(ipl-module-macro_placer 
    PUBLIC
        ipl-module-logger
        ipl-solver-partition 
        ipl-solver-simulate_anneal
        ipl-configurator
)

