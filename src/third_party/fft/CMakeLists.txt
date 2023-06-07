add_library(fftsg_library
                fftsg.cpp
                fftsg2d.cpp)

set(FFTSG_HOME ${HOME_THIRDPARTY}/fft)
target_include_directories(fftsg_library PUBLIC ${FFTSG_HOME})