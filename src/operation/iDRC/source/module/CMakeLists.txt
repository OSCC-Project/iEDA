if(DEBUG_IDRC_MODULE)            
    message(STATUS "DRC: DEBUG_IDRC_MODULE")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_MODULE")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_module
    ${IDRC_MODULE}/rule/AdjacentCutSpacing.cpp
    ${IDRC_MODULE}/rule/CornerFillSpacing.cpp
    ${IDRC_MODULE}/rule/CutEOLSpacing.cpp
    ${IDRC_MODULE}/rule/CutShort.cpp
    ${IDRC_MODULE}/rule/DifferentLayerCutSpacing.cpp
    ${IDRC_MODULE}/rule/Enclosure.cpp
    ${IDRC_MODULE}/rule/EnclosureEdge.cpp
    ${IDRC_MODULE}/rule/EnclosureParallel.cpp
    ${IDRC_MODULE}/rule/EndOfLineSpacing.cpp
    ${IDRC_MODULE}/rule/FloatingPatch.cpp
    ${IDRC_MODULE}/rule/JogToJogSpacing.cpp
    ${IDRC_MODULE}/rule/MaxViaStack.cpp
    ${IDRC_MODULE}/rule/MetalShort.cpp
    ${IDRC_MODULE}/rule/MinHole.cpp
    ${IDRC_MODULE}/rule/MinimumArea.cpp
    ${IDRC_MODULE}/rule/MinimumCut.cpp
    ${IDRC_MODULE}/rule/MinimumWidth.cpp
    ${IDRC_MODULE}/rule/MinStep.cpp
    ${IDRC_MODULE}/rule/NonsufficientMetalOverlap.cpp
    ${IDRC_MODULE}/rule/NotchSpacing.cpp
    ${IDRC_MODULE}/rule/OffGridOrWrongWay.cpp
    ${IDRC_MODULE}/rule/OutOfDie.cpp
    ${IDRC_MODULE}/rule/ParallelRunLengthSpacing.cpp
    ${IDRC_MODULE}/rule/SameLayerCutSpacing.cpp
)

target_link_libraries(idrc_module 
    PUBLIC
        idrc_data_manager
)

target_include_directories(idrc_module 
    PUBLIC
        ${IDRC_MODULE}
)
