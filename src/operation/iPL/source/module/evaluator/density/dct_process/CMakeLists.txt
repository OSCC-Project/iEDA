add_library(ipl-dct 
            DCT.cc
            FFT.cc)

target_link_libraries(ipl-dct 
    PUBLIC
        ipl-bridge)