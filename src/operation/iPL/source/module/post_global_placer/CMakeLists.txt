add_library(ipl-module-post_gp
    PostGP.cc
)

target_link_libraries(ipl-module-post_gp
    PUBLIC
        ipl-configurator
        ipl-module-evaluator
        ipl-module-legalizer
        ipl-bridge
)

target_include_directories(ipl-module-post_gp PUBLIC ${iPL_POSTGP})