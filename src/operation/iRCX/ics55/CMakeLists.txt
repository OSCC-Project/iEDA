include(GNUInstallDirs)

# find_package(IrcxIcs55 CONFIG REQUIRED)
set(_IRCX_ICS55_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
set(_IRCX_ICS55_LIBRARY "${CMAKE_CURRENT_LIST_DIR}/lib/libircx_ics55.so")

if(NOT EXISTS "${_IRCX_ICS55_INCLUDE_DIR}/ircx_ics55.h")
  message(FATAL_ERROR "iRCX ICS55 repo-local SDK header is missing: ${_IRCX_ICS55_INCLUDE_DIR}/ircx_ics55.h")
endif()
if(NOT EXISTS "${_IRCX_ICS55_LIBRARY}")
  message(FATAL_ERROR "iRCX ICS55 repo-local SDK library is missing: ${_IRCX_ICS55_LIBRARY}")
endif()

# get_target_property(_IRCX_ICS55_INCLUDE_DIR IrcxIcs55::ircx_ics55 INTERFACE_INCLUDE_DIRECTORIES)
# get_target_property(_IRCX_ICS55_LIBRARY IrcxIcs55::ircx_ics55 IMPORTED_LOCATION)

# file(WRITE "${CMAKE_BINARY_DIR}/ircx_ics55_library.txt" "${_IRCX_ICS55_LIBRARY}\n")

find_package(OpenMP REQUIRED)

set(_IRCX_ICS55_LINK_LIBRARIES
  OpenMP::OpenMP_CXX
)

add_library(ircx_ics55 SHARED IMPORTED GLOBAL)

set_target_properties(ircx_ics55
  PROPERTIES
    IMPORTED_LOCATION "${_IRCX_ICS55_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_IRCX_ICS55_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${_IRCX_ICS55_LINK_LIBRARIES}"
)

install(FILES "${_IRCX_ICS55_LIBRARY}"
  DESTINATION ${ECC_INSTALL_LIB_DIR}
)
