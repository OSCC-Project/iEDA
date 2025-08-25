
# set(CMAKE_BUILD_TYPE "Debug")

include_directories(${iPL_CONFIG})

add_library(ipl-module-detail_placer
    database/DPCell.cc
    database/DPCluster.cc
    database/DPDatabase.cc
    database/DPDesign.cc
    database/DPInstance.cc
    database/DPInterval.cc
    database/DPLayout.cc
    database/DPNet.cc
    database/DPPin.cc
    database/DPRegion.cc
    database/DPRow.cc
    database/DPBin.cc
    database/DPSegment.cc
    database/DPNode.cc

    operation/BinOpt.cc
    operation/InstanceSwap.cc
    operation/LocalReorder.cc
    operation/RowOpt.cc
    operation/NFSpread.cc

    DPOperator.cc
    DetailPlacer.cc
)

target_link_libraries(ipl-module-detail_placer
    PUBLIC
            ipl-configurator
            ipl-module-grid_manager
            ipl-module-topology_manager
            ipl_module_evaluator_wirelength
            )

target_include_directories(ipl-module-detail_placer PUBLIC ${iPL_DP})
