SET(IPLF_HOME ${HOME_PLATFORM})

SET(IUTILITY_HOME ${HOME_UTILITY})
SET(THIRD_PARTY_HOME ${HOME_THIRDPARTY})

SET(ISTA_HOME ${HOME_OPERATION}/iSTA)
SET(IRT_HOME ${HOME_OPERATION}/iRT)

SET(IEVAL_HOME ${HOME_EVALUATION})
SET(IEVAL_SOURCE ${IEVAL_HOME}/source)
SET(IEVAL_DATA ${IEVAL_HOME}/data)
SET(IEVAL_MODULE ${IEVAL_SOURCE}/module)

set( IEVAL_HEADER
    ## config
    ${IEVAL_SOURCE}/config
    ## util
    ${IEVAL_SOURCE}/util
    ## congestion_evaluator
    ${IEVAL_MODULE}/congestion
    ## wirelength_evaluator
    ${IEVAL_MODULE}/wirelength
    ## timing_evaluator
    ${IEVAL_MODULE}/timing
    ## gds wrapper
    ${IEVAL_MODULE}/gds_wrapper
    ## wrapper
    ${IEVAL_MODULE}/wrapper
    ${IEVAL_MODULE}/wrapper/database
    ## Evaluator
    ${IEVAL_SOURCE}
    ## data
    ${IEVAL_DATA}
    ## iDB
    ${HOME_DATABASE}/basic/geometry
    ${HOME_DATABASE}/tool_db/design
    ${HOME_DATABASE}/tool_db/design/db_design
    ${HOME_DATABASE}/tool_db/design/db_layout
    ${HOME_DATABASE}/manager/builder
    ${HOME_DATABASE}/manager/builder/def_builder
    ${HOME_DATABASE}/manager/builder/lef_builder
    ${HOME_DATABASE}/manager/service/def_service
    ${HOME_DATABASE}/manager/service/lef_service
    ## iSTA
    ${HOME_UTILITY}/stdBase
    ${HOME_UTILITY}/stdBase/include
    ${HOME_UTILITY}
    ${HOME_OPERATION}/iSTA
    ${HOME_OPERATION}/iSTA/source
    ${HOME_OPERATION}/iSTA/source/module
    ${HOME_OPERATION}/iSTA/source/module/include
    ## iPlatform
    ${IPLF_HOME}/tool_manager
    ${IPLF_HOME}/tool_manager/tool_api
    ${IPLF_HOME}/data_manager
)
INCLUDE_DIRECTORIES(${IEVAL_HEADER})
INCLUDE_DIRECTORIES(SYSTEM
    ${HOME_THIRDPARTY}
    ${HOME_OPERATION}/iSTA/source/third-party
)
