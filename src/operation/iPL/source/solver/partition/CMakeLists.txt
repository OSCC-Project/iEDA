add_library(ipl-solver-partition Metis.cc Hmetis.cc)
target_link_libraries(ipl-solver-partition ${HOME_THIRDPARTY}/hmetis/libmetis.a)
