if(DEBUG_IDRC_RULE_VALIDATOR)
    message(STATUS "DRC: DEBUG_IDRC_RULE_VALIDATOR")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "DRC: RELEASE_IDRC_RULE_VALIDATOR")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(idrc_rule_validator
    ## module rule_validator
    ${IDRC_MODULE}/rule_validator/RuleValidator.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/AdjacentCutSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/CornerFillSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/CornerSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/CutEOLSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/CutShort.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/DifferentLayerCutSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/Enclosure.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/EnclosureEdge.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/EnclosureParallel.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/EndOfLineSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/FloatingPatch.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/JogToJogSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MaximumWidth.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MaxViaStack.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MetalShort.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MinHole.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MinimumArea.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MinimumCut.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MinimumWidth.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/MinStep.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/NonsufficientMetalOverlap.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/NotchSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/OffGridOrWrongWay.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/OutOfDie.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/ParallelRunLengthSpacing.cpp
    ${IDRC_MODULE}/rule_validator/rv_design_rule/SameLayerCutSpacing.cpp
)

target_link_libraries(idrc_rule_validator
    PUBLIC
    idrc_data_manager
    idrc_module
    idrc_toolkit
)

target_include_directories(idrc_rule_validator
    PUBLIC
    ${IDRC_MODULE}/rule_validator/rv_data_manager
    ${IDRC_MODULE}/rule_validator
)
