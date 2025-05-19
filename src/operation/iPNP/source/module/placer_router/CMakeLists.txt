add_library(placer_router
    FastPlacer.cpp
    FastRouter.cpp
)

target_link_libraries(placer_router PUBLIC ipl-api irt_interface)