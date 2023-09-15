option(OPENDACS_2023 "If ON, run contest of OpenDACS 2003 porject." ON)

if(OPENDACS_2023)
    add_subdirectory(OpenDACS_2023/code)
endif()
