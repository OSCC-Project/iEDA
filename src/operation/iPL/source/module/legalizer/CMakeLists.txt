include_directories(${iPL_LG})

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
    PRIVATE
        ipl-module-logger
        ipl-bridge
        ipl-configurator
        solver_legalization

)

target_include_directories(ipl-module-legalizer PUBLIC ${iPL_LG})
