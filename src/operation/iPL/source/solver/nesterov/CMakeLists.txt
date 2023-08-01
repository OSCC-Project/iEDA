add_library(ipl-solver-nesterov Nesterov.cc)

target_link_libraries(ipl-solver-nesterov
    PRIVATE
    ipl-bridge
    ipl-module-logger
)

target_include_directories(ipl-solver-nesterov PUBLIC ${NESTEROV_SOLVER})