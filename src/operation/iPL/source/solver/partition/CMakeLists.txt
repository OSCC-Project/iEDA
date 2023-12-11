add_library(ipl-solver-partition Metis.cc Hmetis.cc)
target_include_directories(ipl-solver-partition PUBLIC ${HOME_THIRDPARTY}/metis)
target_link_libraries(ipl-solver-partition ${HOME_THIRDPARTY}/hmetis/libmetis.a)
