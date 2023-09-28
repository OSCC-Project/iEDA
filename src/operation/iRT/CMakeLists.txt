message(STATUS "")
message(STATUS "############## RT: START CMAKE ##############")

############################ cmake flag ############################

set(CMAKE_CXX_STANDARD 20)

# ADD_COMPILE_OPTIONS("-fsanitize=address")
# LINK_LIBRARIES("-fsanitize=address")

############################ debug api ############################

# set(DEBUG_IRT_API ON)

############################ debug source module ############################

# set(DEBUG_IRT_DETAILED_ROUTER ON)
# set(DEBUG_IRT_DRC_CHECKER ON)
# set(DEBUG_IRT_EARLY_GLOBAL_ROUTER ON)
# set(DEBUG_IRT_GDS_PLOTTER ON)
# set(DEBUG_IRT_GLOBAL_ROUTER ON)
# set(DEBUG_IRT_PIN_ACCESSOR ON)
# set(DEBUG_IRT_REGION_QUERY ON)
# set(DEBUG_IRT_RESOURCE_ALLOCATOR ON) 
# set(DEBUG_IRT_TRACK_ASSIGNER ON)
# set(DEBUG_IRT_VIOLATION_REPAIRER ON)

############################ debug source solver ############################

# set(DEBUG_IRT_UNIVERSAL_ROUTER ON)

############################ debug source data_manager ############################

# set(DEBUG_IRT_DATA_MANAGER ON)

############################ debug source utility ############################

# set(DEBUG_IRT_LOGGER ON)
# set(DEBUG_IRT_MONITOR ON)
# set(DEBUG_IRT_REPORTER ON)

############################ setting path ############################

set(HOME_IRT ${HOME_OPERATION}/iRT)
set(IRT_API ${HOME_IRT}/api)
set(IRT_EXTERNAL_LIBS ${HOME_IRT}/external_libs)
set(IRT_SOURCE ${HOME_IRT}/source)
set(IRT_TEST ${HOME_IRT}/test)

############################ cmake subdirectory ############################

add_subdirectory(${IRT_API})
add_subdirectory(${IRT_EXTERNAL_LIBS})
add_subdirectory(${IRT_SOURCE})
# add_subdirectory(${IRT_TEST})

########################################################

message(STATUS "############## RT: END CMAKE ##############\n")
