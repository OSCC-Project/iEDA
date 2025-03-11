add_library(idrc_condition_manager
  condition_manager.cpp
  condition_min_spacing.cpp
  condition_overlap.cpp
  condition_jog.cpp
  condition_polygon.cpp
  condition_paralle_length.cpp
  condition_area.cpp
  condition_cut.cpp
)

target_link_libraries(idrc_condition_manager
    PUBLIC
    idrc_pro_util
    idrc_pro_violation
    idrc_pro_tech_rules
    idrc_engine_manager
    idrc_pro_data
)

target_include_directories(idrc_condition_manager
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DEBUG_IDRC_CONDITION)
  message(STATUS "iDRC: DEBUG_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Debug")
  # set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -Wextra -Werror=return-type")
else()
  message(STATUS "iDRC: RELEASE_IDRC_CONDITION")
  set(CMAKE_BUILD_TYPE "Release")
endif()
