
set(CTS_PATH ${HOME_OPERATION}/iCTS)
set(CTS_SRC_PATH ${CTS_PATH}/source)

set(ICTS_HEADER
    ${CTS_SRC_PATH}
    ${CTS_SRC_PATH}/solver/DME/include
    ${CTS_SRC_PATH}/solver/polygon/include
    ${CTS_SRC_PATH}/solver/clustering/include
    ${CTS_SRC_PATH}/config
    ${CTS_SRC_PATH}/database
    ${CTS_SRC_PATH}/io
    ${CTS_SRC_PATH}/tcl
    ${CTS_SRC_PATH}/util
    ${CTS_SRC_PATH}/util/geometry
    ${CTS_SRC_PATH}/util/plot
    ${CTS_SRC_PATH}/module
    ${CTS_SRC_PATH}/module/evaluator
    ${CTS_SRC_PATH}/module/evaluator/data
    ${CTS_SRC_PATH}/module/synthesis
    ${CTS_SRC_PATH}/module/synthesis/data
    ${CTS_SRC_PATH}/module/synthesis/operator
    ${CTS_SRC_PATH}/module/optimizer
    ${CTS_SRC_PATH}/module/balancer
    ${CTS_SRC_PATH}/module/router
)

# include_directories(${ICTS_HEADER})
