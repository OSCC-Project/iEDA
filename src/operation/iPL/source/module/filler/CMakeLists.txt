add_library(ipl-module-filler src/MapFiller.cpp)

target_link_libraries(ipl-module-filler
    PUBLIC
        ipl-configurator
)   

target_include_directories(ipl-module-filler PUBLIC ${iPL_FILLER})