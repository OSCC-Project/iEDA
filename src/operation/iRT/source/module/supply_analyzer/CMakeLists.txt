if(DEBUG_IRT_SUPPLY_ANALYZER)
    message(STATUS "RT: DEBUG_IRT_SUPPLY_ANALYZER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_SUPPLY_ANALYZER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_supply_analyzer
    ## module supply_analyzer
    ${IRT_MODULE}/supply_analyzer/SupplyAnalyzer.cpp
)

target_link_libraries(irt_supply_analyzer
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_supply_analyzer
    PUBLIC
    ${IRT_MODULE}/supply_analyzer/sa_data_manager
    ${IRT_MODULE}/supply_analyzer

)
