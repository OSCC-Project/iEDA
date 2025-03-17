message(STATUS "")
message(STATUS "############## DRC: STADRC CMAKE ##############")

############################ cmake flag ############################

set(CMAKE_CXX_STANDARD 20)

# ADD_COMPILE_OPTIONS("-Wconversion")
# ADD_COMPILE_OPTIONS("-fsanitize=address")
# LINK_LIBRARIES("-fsanitize=address")

############################ debug interface ############################

# set(DEBUG_IDRC_INTERFACE ON)

############################ debug source data_manager ############################

# set(DEBUG_IDRC_DATA_MANAGER ON)

############################ debug source module ############################

# set(DEBUG_IDRC_RULE_VALIDATOR ON)
# set(DEBUG_IDRC_GDS_PLOTTER ON)

############################ debug source utility ############################

# set(DEBUG_IDRC_LOGGER ON)
# set(DEBUG_IDRC_MONITOR ON)
# set(DEBUG_IDRC_UTILITY ON)

############################ setting path ############################

set(HOME_IDRC ${HOME_OPERATION}/iDRC)

set(IDRC_INTERFACE ${HOME_IDRC}/interface)
set(IDRC_SOURCE ${HOME_IDRC}/source)

############################ cmake subdirectory ############################

add_subdirectory(${IDRC_INTERFACE})
add_subdirectory(${IDRC_SOURCE})

########################################################

message(STATUS "############## DRC: END CMAKE ##############\n")
