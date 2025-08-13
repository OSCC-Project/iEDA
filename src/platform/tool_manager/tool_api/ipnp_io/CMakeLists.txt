add_library(tool_api_ipnp
    ipnp_io.cpp
)

target_include_directories(tool_api_ipnp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${HOME_OPERATION}/iPNP/source                    
    ${HOME_OPERATION}/iPNP/source/config             
    ${HOME_OPERATION}/iPNP/source/include            
    ${HOME_OPERATION}/iPNP/source/data_manager       
    ${HOME_OPERATION}/iPNP/source/module             
    ${HOME_OPERATION}/iPNP/source/module/evaluator  
    ${HOME_OPERATION}/iPNP/source/module/optimizer   
    ${HOME_OPERATION}/iPNP/source/module/synthesis   
    ${HOME_OPERATION}/iPNP/api                       
)

target_link_libraries(tool_api_ipnp
    PRIVATE
        flow_config
        ipnp_api
)
