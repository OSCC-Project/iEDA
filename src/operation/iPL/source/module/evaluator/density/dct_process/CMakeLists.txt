add_library(ipl-dct 
            DCT.cc)

target_link_libraries(ipl-dct 
    PUBLIC
        ipl-utility)