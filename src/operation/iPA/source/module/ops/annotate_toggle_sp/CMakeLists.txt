cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 20)

aux_source_directory(./ SRC)
add_library(annotate ${SRC})

set(LIBGMP "gmp")

if(BUILD_STATIC_LIB)
    find_library(GMP_STATIC_LIB 
      NAMES libgmp.a
      PATHS /usr/lib/x86_64-linux-gnu
      NO_DEFAULT_PATH
    )
    if(GMP_STATIC_LIB)
        set(LIBGMP ${GMP_STATIC_LIB})
    endif()
endif()

target_link_libraries(annotate ${LIBGMP} core)

