include(${HOME_CMAKE}/operation/idb.cmake)
include(${HOME_CMAKE}/operation/iplf.cmake)

INCLUDE_DIRECTORIES(
    SYSTEM
    ## Third Party
    ${HOME_THIRDPARTY}/json
    ## Boost
    ${Boost_INCLUDE_DIRS} 
)

add_subdirectory(api)
add_subdirectory(source)
# add_subdirectory(test)