include(${HOME_CMAKE}/operation/ista.cmake)
include(${HOME_CMAKE}/operation/ipa.cmake)

add_subdirectory(builder)
add_subdirectory(memory)
add_subdirectory(parser)
add_subdirectory(service)

# SET(CMAKE_BUILD_TYPE "Debug")