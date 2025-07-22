if(NOT DEFINED CMD_BUILD)
    # set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Congestion Evaluation APP
add_executable(congestion_app
    congestion_app.cpp
)

target_link_libraries(congestion_app
    PRIVATE
    eval_congestion_api
    idm
)

target_include_directories(congestion_app
    PRIVATE
    ${HOME_EVALUATION}/api
    ${HOME_EVALUATION}/database
)

# Density Evaluation APP
add_executable(density_app
    density_app.cpp
)

target_link_libraries(density_app
    PRIVATE
    eval_density_api
    idm
)

target_include_directories(density_app
    PRIVATE
    ${HOME_EVALUATION}/api
    ${HOME_EVALUATION}/database
)

# Timing & Power Evaluation APP
add_executable(timing_app
    timing_app.cc
)

target_link_libraries(timing_app
    PRIVATE
    eval_timing_api
    idm
    -u_ZN3ipl17SteinerWirelength25updateAllNetWorkPointPairEv
    -u_ZN3ipl13NesterovPlace21printNesterovDatabaseEv
    -u_ZN3ito3iTO5runTOEv
    -u_ZN3idm11DataManager7saveDefENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
    -u_ZN4icts9GDSPloter10plotDesignERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
)

target_include_directories(timing_app
    PRIVATE
    ${HOME_EVALUATION}/api
    ${HOME_EVALUATION}/database
)

# Wirelength Evaluation APP
add_executable(wirelength_app
    wirelength_app.cpp
)

target_link_libraries(wirelength_app
    PRIVATE
    eval_wirelength_api
    idm
)

target_include_directories(wirelength_app
    PRIVATE
    ${HOME_EVALUATION}/api
    ${HOME_EVALUATION}/database
)

# Union Evaluation APP
add_executable(union_app
    union_app.cpp
)

target_link_libraries(union_app
    PRIVATE
    idm
    eval_wirelength_api
    eval_timing_api
    eval_congestion_api
    eval_union_api
)

target_include_directories(union_app
    PRIVATE
    ${HOME_EVALUATION}/api
    ${HOME_EVALUATION}/database
)

