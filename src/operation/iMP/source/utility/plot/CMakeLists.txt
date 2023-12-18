add_library(imp-plot
    ${IMP_UTILITY}/plot/PyPlot.cc
)

target_include_directories(imp-plot
    INTERFACE
        ${IMP_UTILITY}/plot
)
message(STATUS "MP: Build libimp-plot.")
