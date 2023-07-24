add_library(ipl-module-wrapper
            IDBWrapper.cc)

target_link_libraries(ipl-module-wrapper
    PRIVATE
        ipl-bridge
        ipl-module-logger
)

target_include_directories(ipl-module-wrapper PUBLIC ${iPL_WRAPPER})