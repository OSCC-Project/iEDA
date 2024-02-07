
# set(CMAKE_BUILD_TYPE "Debug")

include_directories(${iPL_CONFIG})

# set(Boost_INCLUDE_DIR /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0/)
# find_package(Boost COMPONENTS system filesystem program_options REQUIRED)


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
            # ${Boost_LIBRARIES}
            )

target_include_directories(ipl-module-detail_placer PUBLIC ${iPL_DP})

# Boost   
# target_include_directories(ipl-module-detail_placer PUBLIC ${Boost_INCLUDE_DIR})
