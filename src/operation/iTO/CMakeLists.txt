set(CMAKE_CXX_STANDARD 20)

message(STATUS "")
message(STATUS "=================  TO: Debugging Options  =================")
# =================  Debugging Options  =================
# api
# set(DEBUG_ITO_API ON)

# source/config
# set(DEBUG_ITO_CONFIG ON)

# source/data_manager
# set(DEBUG_ITO_DATA_MANAGER ON)

# source/module/evaluator
# set(DEBUG_ITO_EVALUATOR ON)

# source/module/fix_drv
# set(DEBUG_ITO_FIX_DRV ON)

# source/module/fix_hold
# set(DEBUG_ITO_FIX_HOLD ON)

# source/module/fix_setup
# set(DEBUG_ITO_FIX_SETUP ON)

# source/module/placer
# set(DEBUG_ITO_PLACER ON)

# source/module/solver/tree_build
# set(DEBUG_ITO_TREE_BUILD ON)
# set(DEBUG_ITO_TREE_BUILD_HVTREE ON)
# set(DEBUG_ITO_TREE_BUILD_FLUTE ON)
# set(DEBUG_ITO_TREE_BUILD_SLT ON)


# source/module/solver/buffer/vg
# set(DEBUG_ITO_VG_BUFFER ON)

# source/timing_engine
# set(DEBUG_ITO_TIMING_ENGINE ON)

# source/utility
# set(DEBUG_ITO_UTILITY ON)
# =================  Debugging Options  =================


add_subdirectory(api)
add_subdirectory(source)

message(STATUS "=================  TO: Debugging Options  =================")
message(STATUS "")