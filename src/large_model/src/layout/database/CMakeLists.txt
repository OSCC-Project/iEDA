add_library(ilm_layout_db
lm_layout.cpp
lm_patch.cpp
)

target_include_directories(ilm_layout_db 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)