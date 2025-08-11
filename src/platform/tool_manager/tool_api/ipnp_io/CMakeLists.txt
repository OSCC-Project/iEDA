add_library(tool_api_ipnp
    ipnp_io.cpp
)

target_include_directories(tool_api_ipnp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    # 只为这个库添加iPNP相关路径，不影响其他库
    ${HOME_OPERATION}/iPNP/source                    # iPNP.hh
    ${HOME_OPERATION}/iPNP/source/config             # PNPConfig.hh
    ${HOME_OPERATION}/iPNP/source/include            # iPNPCommon.hh等
    ${HOME_OPERATION}/iPNP/source/data_manager       # GridManager.hh等
    ${HOME_OPERATION}/iPNP/source/module             # 模块头文件
    ${HOME_OPERATION}/iPNP/source/module/evaluator   # 评估器头文件
    ${HOME_OPERATION}/iPNP/source/module/optimizer   # 优化器头文件
    ${HOME_OPERATION}/iPNP/source/module/synthesis   # PowerRouter.hh等
    ${HOME_OPERATION}/iPNP/api                       # iPNPApi.hh
)

target_link_libraries(tool_api_ipnp
    PRIVATE
        flow_config
        idm
        iPNPApi
        pnp
)
