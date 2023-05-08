
add_library(ipdn_plan
    pdn_plan.cpp
    pdn_plan_macro.cpp
    pdn_cut_stripe.cpp
)

target_link_libraries(ipdn_plan 
    PUBLIC
    idm
    ipdn_db
    ipdn_via
)

target_include_directories(ipdn_plan 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iPDN/source/data
    ${HOME_OPERATION}/iPDN/source/module/pdn_via
)