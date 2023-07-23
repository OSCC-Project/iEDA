add_library(imp-logger
    ${IMP_UTILITY}/logger/Logger.cpp
)

target_include_directories(imp-logger 
    PUBLIC
        ${IMP_UTILITY}/logger
)
