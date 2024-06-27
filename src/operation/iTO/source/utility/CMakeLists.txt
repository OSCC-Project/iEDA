add_library(ito_utility 
    Utility.cpp
    GDSwriter.cpp
    Reporter.cpp
)

target_link_libraries(ito_utility 
PUBLIC
        idb
        idm
        ista-engine
)

target_include_directories(ito_utility 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_OPERATION}/iTO/source/data
)
