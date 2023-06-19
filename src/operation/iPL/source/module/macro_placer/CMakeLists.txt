include_directories(${iPL_MP})

add_library(ipl-module-macro_placer
    database/FPRect.cc
    database/FPInst.cc
    database/FPNet.cc
    database/FPPin.cc
    database/FPDesign.cc
    database/FPLayout.cc
    gds_plotter/GDSPlotter.cc

    MacroPlacer.cc
    MPDB.cc
    Setting.cc
    wrapper/IPLDBWrapper.cc
    partition/MPPartition.cc
    partition/HierPartition.cc
    partition/Module.cc
    simulate_anneal/BStarTree.cc
    simulate_anneal/SequencePair.cc
    simulate_anneal/MPEvaluation.cc
    analytical_method/Solver.cc
    analytical_method/Problem.cc
    analytical_method/MProblem.cc
    analytical_method/LSEWirelength.cc)

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
    ipl-utility
)
