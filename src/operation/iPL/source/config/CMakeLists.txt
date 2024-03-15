add_library(ipl-configurator Configurator.cc)

target_include_directories(ipl-configurator
    PUBLIC
    ${iPL_CONFIG}
    ${iPL_SOURCE}/module/buffer/config
    ${iPL_SOURCE}/module/detail_placer/config
    ${iPL_SOURCE}/module/filler/config
    ${iPL_SOURCE}/module/global_placer/electrostatic_placer/config
    ${iPL_SOURCE}/module/legalizer/config
    ${iPL_SOURCE}/module/macro_placer/config
    ${iPL_SOURCE}/module/post_global_placer/config
)

target_include_directories(ipl-configurator PUBLIC ${iPL_CONFIG})