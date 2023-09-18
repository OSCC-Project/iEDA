include_directories(${iPL_LG})

set(LG_METHOD ${iPL_LG}/method)
add_subdirectory(${LG_METHOD})

add_library(ipl-module-legalizer 
    database/LGCell.cc
    database/LGDatabase.cc
    database/LGInstance.cc
    database/LGLayout.cc
    database/LGRegion.cc
    database/LGRow.cc
    database/LGInterval.cc
    Legalizer.cc
)

target_link_libraries(ipl-module-legalizer
    PUBLIC
        legalizer_method
    PRIVATE
        ipl-module-logger
        ipl-bridge
        ipl-configurator

)

target_include_directories(ipl-module-legalizer PUBLIC ${iPL_LG})
