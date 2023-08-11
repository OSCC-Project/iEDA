if(DEBUG_ITO_MODULE)            
    message(STATUS "TO: DEBUG_ITO_MODULE")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_MODULE")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_module 
        ${ITO_MODULE_PATH}/evaluator/RoutingTree.cpp
        ${ITO_MODULE_PATH}/evaluator/EstimateParasitics.cpp  
        
        ${ITO_MODULE_PATH}/fix_hold/HoldOptimizer.cpp 
        ${ITO_MODULE_PATH}/fix_drv/ViolationOptimizer.cpp
        ${ITO_MODULE_PATH}/fix_drv/CTSViolationFixer.cpp
        ${ITO_MODULE_PATH}/fix_drv/FixWireLength.cpp
        ${ITO_MODULE_PATH}/fix_setup/BufferedOption.cpp
        ${ITO_MODULE_PATH}/fix_setup/SetupOptimizer.cpp
        
        ${ITO_MODULE_PATH}/placer/Placer.cpp
)

target_link_libraries(ito_module 
    PUBLIC
        ito_api
        ito_source_external_libs
)

target_include_directories(ito_module 
    PUBLIC
        ${ITO_MODULE_PATH}
        ${ITO_PATH}/api
        ${ITO_PATH}/source/utility
        # module
        ${ITO_SOURCE_PATH}/module
        ${ITO_SOURCE_PATH}/module/evaluator
        ${ITO_SOURCE_PATH}/module/fix_drv
        ${ITO_SOURCE_PATH}/module/fix_hold
        ${ITO_SOURCE_PATH}/module/fix_setup
        ${ITO_SOURCE_PATH}/module/placer
        ${ITO_DATA_PATH}
        ${ITO_IO_PATH}
        ${HOME_THIRDPARTY}/parser/liberty/
        ${HOME_DATABASE}/manager/parser/liberty
)
