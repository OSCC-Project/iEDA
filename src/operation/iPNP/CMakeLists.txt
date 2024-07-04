include(${HOME_CMAKE}/operation/idb.cmake)
include(${HOME_CMAKE}/operation/iplf.cmake)

add_subdirectory(api)
add_subdirectory(data)
add_subdirectory(source)
add_subdirectory(test)

add_executable(iPNP main.cpp)