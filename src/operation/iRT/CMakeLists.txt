message(STATUS "")
message(STATUS "############## RT: START CMAKE ##############")

############################ cmake flag ############################

set(CMAKE_CXX_STANDARD 20)

# add_compile_options("-Wconversion")
# add_compile_options("-fsanitize=address")
# link_libraries("-fsanitize=address")

############################ debug interface ############################

# set(DEBUG_IRT_INTERFACE ON)

############################ debug source data_manager ############################

# set(DEBUG_IRT_DATA_MANAGER ON)

############################ debug source module ############################

# set(DEBUG_IRT_DRC_ENGINE ON)
# set(DEBUG_IRT_GDS_PLOTTER ON)
# set(DEBUG_IRT_PIN_ACCESSOR ON)
# set(DEBUG_IRT_SUPPLY_ANALYZER ON) 
# set(DEBUG_IRT_TOPOLOGY_GENERATOR ON)
# set(DEBUG_IRT_LAYER_ASSIGNER ON)
# set(DEBUG_IRT_SPACE_ROUTER ON)
# set(DEBUG_IRT_TRACK_ASSIGNER ON) 
# set(DEBUG_IRT_DETAILED_ROUTER ON)
# set(DEBUG_IRT_VIOLATION_REPORTER ON)
# set(DEBUG_IRT_EARLY_ROUTER ON)

############################ debug source utility ############################

# set(DEBUG_IRT_LOGGER ON)
# set(DEBUG_IRT_MONITOR ON)
# set(DEBUG_IRT_UTILITY ON)

############################ setting path ############################

set(HOME_IRT ${HOME_OPERATION}/iRT)

set(IRT_INTERFACE ${HOME_IRT}/interface)
set(IRT_SOURCE ${HOME_IRT}/source)
set(IRT_TEST ${HOME_IRT}/test)

############################ cmake subdirectory ############################

add_subdirectory(${IRT_INTERFACE})
add_subdirectory(${IRT_SOURCE})
add_subdirectory(${IRT_TEST})

########################################################

message(STATUS "############## RT: END CMAKE ##############\n")
