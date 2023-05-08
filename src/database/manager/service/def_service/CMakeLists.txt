if(BUILD_STATIC_LIB)
  add_library( def_service   
  def_service.cpp
  )
else()
  add_library( def_service SHARED
  def_service.cpp
  )
endif()

target_link_libraries(def_service PRIVATE  idb)

target_include_directories(def_service 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${HOME_DATABASE}/manager/service/lef_service
)