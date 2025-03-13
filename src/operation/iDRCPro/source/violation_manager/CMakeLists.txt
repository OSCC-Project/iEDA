add_library(idrc_pro_violation
    idrc_violation_manager.cpp
    idrc_violation.cpp
)

target_link_libraries(idrc_pro_violation
    PUBLIC
    solver_geometry_boost
    idrc_pro_tech_rules
    idm
    idrc_engine_manager
)

target_include_directories(idrc_pro_violation
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)
if(DEBUG_IDRC_VIOLATION)
  message(STATUS "iDRC: DEBUG_IDRC_VIOLATION")
  set(CMAKE_BUILD_TYPE "Debug")
  # set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -Wextra -Werror=return-type")
else()
  message(STATUS "iDRC: RELEASE_IDRC_VIOLATION")
  set(CMAKE_BUILD_TYPE "Release")
endif()