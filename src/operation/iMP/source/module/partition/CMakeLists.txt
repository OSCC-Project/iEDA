add_library(imp-partition 
            ${IMP_PARTITION}/Hmetis.cc
            ${IMP_PARTITION}/MtKahypar.cc)
target_link_libraries(imp-partition
                      PRIVATE
                      imp-logger
                      mtkahypar
                      )

target_include_directories(imp-partition
                           PUBLIC
                           ${IMP_PARTITION})

set(HMETIS_BINARY ${PROJECT_SOURCE_DIR}/src/third_party/hmetis/hmetis2.0pre1)
message(STATUS "MP: Hmetis binary path - " ${HMETIS_BINARY})
add_definitions("-DHMETIS_BINARY=\"${HMETIS_BINARY}\"")

add_executable(imp-test-partition partition_example.cc)
target_link_libraries(imp-test-partition PRIVATE imp-partition)