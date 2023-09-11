add_subdirectory(${ICTS_DATA_MANAGER}/config)
add_subdirectory(${ICTS_DATA_MANAGER}/database)
add_subdirectory(${ICTS_DATA_MANAGER}/io)

add_library(icts_data_manager INTERFACE)

target_link_libraries(icts_data_manager INTERFACE icts_config icts_database
                                                  icts_io icts_module)

target_include_directories(
  icts_data_manager
  INTERFACE ${ICTS_DATA_MANAGER}/config ${ICTS_DATA_MANAGER}/database
            ${ICTS_DATA_MANAGER}/io)
