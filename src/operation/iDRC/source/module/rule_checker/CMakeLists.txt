if(DEBUG_IDRC_RULE_CHECKER)
    message(STATUS "DRC: DEBUG_IDRC_RULE_CHECKER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_RULE_CHECKER")   
    set(CMAKE_BUILD_TYPE "Release")   
endif()

add_library(idrc_rule_checker
    ## module rule_checker
    ${IDRC_MODULE}/rule_checker/RuleChecker.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/AdjacentCutSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/CornerFillSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/CutEOLSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/CutShort.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/DifferentLayerCutSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/Enclosure.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/EnclosureEdge.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/EnclosureParallel.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/EndOfLineSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/FloatingPatch.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/JogToJogSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MaxViaStack.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MetalShort.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MinHole.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MinimumArea.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MinimumCut.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MinimumWidth.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/MinStep.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/NonsufficientMetalOverlap.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/NotchSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/OffGridOrWrongWay.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/OutOfDie.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/ParallelRunLengthSpacing.cpp
    ${IDRC_MODULE}/rule_checker/design_rule/SameLayerCutSpacing.cpp
)

target_link_libraries(idrc_rule_checker 
    PUBLIC
        idrc_data_manager
        idrc_module
        idrc_toolkit
)

target_include_directories(idrc_rule_checker 
    PUBLIC
        ${IDRC_MODULE}/rule_checker/rc_data_manager
        ${IDRC_MODULE}/rule_checker
)
