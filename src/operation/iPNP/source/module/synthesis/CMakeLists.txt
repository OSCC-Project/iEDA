include_directories(${HOME_PLATFORM}/data_manager)

add_library(synthesis
    NetworkSynthesis.cpp
    PowerRouter.cpp
    PowerVia.cpp
)