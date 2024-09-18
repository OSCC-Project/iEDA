## cmake flag
set(CMAKE_CXX_STANDARD 20)

## set path
set(EVAL_PRO_API ${HOME_EVALUATION_PRO}/api)
set(EVAL_PRO_APPS ${HOME_EVALUATION_PRO}/apps)
set(EVAL_PRO_DATA ${HOME_EVALUATION_PRO}/database)
set(EVAL_PRO_SOURCE ${HOME_EVALUATION_PRO}/src)

## build
add_subdirectory(${EVAL_PRO_API})
add_subdirectory(${EVAL_PRO_APPS})
add_subdirectory(${EVAL_PRO_DATA})
add_subdirectory(${EVAL_PRO_SOURCE})
