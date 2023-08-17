find_program(HMETIS_EXECUTABLE hmetis HINTS ${PROJECT_SOURCE_DIR})

if(HMETIS_EXECUTABLE)
   set(HMETIS_FOUND true)
   message(STATUS "Found hmetis: ${HMETIS_EXECUTABLE}")
else()
   set(HMETIS_FOUND false)
endif()

