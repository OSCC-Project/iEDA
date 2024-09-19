## cmake flag
set(CMAKE_CXX_STANDARD 20)

## set path
set(EVAL_API ${HOME_EVALUATION}/api)
set(EVAL_APPS ${HOME_EVALUATION}/apps)
set(EVAL_DATA ${HOME_EVALUATION}/database)
set(EVAL_SOURCE ${HOME_EVALUATION}/src)

## build
add_subdirectory(${EVAL_API})
add_subdirectory(${EVAL_APPS})
add_subdirectory(${EVAL_DATA})
add_subdirectory(${EVAL_SOURCE})
