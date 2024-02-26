add_library(imp-plot
    ${IMP_UTILITY}/plot/PyPlot.cc
    ${IMP_UTILITY}/plot/CairoPlot.cc
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED cairo)

target_link_libraries(imp-plot
   PRIVATE
   ${CAIRO_LIBRARIES}
   )
target_include_directories(imp-plot
    PUBLIC
   ${IMP_UTILITY}/plot
   ${CAIRO_INCLUDE_DIRS}
)
message(STATUS "MP: Build libimp-plot.")
