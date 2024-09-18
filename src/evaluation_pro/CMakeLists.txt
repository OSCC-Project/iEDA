## cmake flag
set(CMAKE_CXX_STANDARD 20)

## set path
set(EVAL_API ${HOME_EVALUATION_PRO}/api)
set(EVAL_APPS ${HOME_EVALUATION_PRO}/apps)
set(EVAL_DATA ${HOME_EVALUATION_PRO}/database)
set(EVAL_SOURCE ${HOME_EVALUATION_PRO}/src)

## build
add_subdirectory(${EVAL_API})
add_subdirectory(${EVAL_APPS})
add_subdirectory(${EVAL_DATA})
add_subdirectory(${EVAL_SOURCE})
