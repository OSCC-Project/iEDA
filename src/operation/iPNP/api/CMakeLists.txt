add_library(ipdn_api
    ipdn_api.cpp

)

target_include_directories(ipdn_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iPDN/source/data
    ${HOME_OPERATION}/iPDN/source/module/pdn_plan
    ${HOME_OPERATION}/iPDN/source/module/pdn_router
    ${HOME_OPERATION}/iPDN/source/module/pdn_sim
    ${HOME_OPERATION}/iPDN/source/module/pdn_via
    
)

target_link_libraries(ipdn_api
    PRIVATE
        tool_manager
        ipdn_plan
        ipdn_via
        
)