## cmake flag
set(CMAKE_CXX_STANDARD 20)

## cmake debug
# set(DEBUG_EVAL_API ON)
# set(DEBUG_EVAL_SOURCE ON)
# set(DEBUG_EVAL_CONFIG ON)
# set(DEBUG_EVAL_CONGESTION ON)
# set(DEBUG_EVAL_DRC ON)
# set(DEBUG_EVAL_GDS_WRAPPER ON)
# set(DEBUG_EVAL_POWER ON)
# set(DEBUG_EVAL_TIMING ON)
# set(DEBUG_EVAL_WIRELENGTH ON)
# set(DEBUG_EVAL_WRAPPER ON)

## set path
set(EVAL_API ${HOME_EVALUATION}/api)
set(EVAL_DATA ${HOME_EVALUATION}/data)
set(EVAL_EXTERNAL ${HOME_EVALUATION}/external)
set(EVAL_SOURCE ${HOME_EVALUATION}/source)
set(EVAL_TEST ${HOME_EVALUATION}/test)

## build
add_subdirectory(${EVAL_API})
add_subdirectory(${EVAL_DATA})
add_subdirectory(${EVAL_EXTERNAL})
add_subdirectory(${EVAL_SOURCE})
add_subdirectory(${EVAL_TEST})
