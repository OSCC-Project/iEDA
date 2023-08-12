# Taken from https://github.com/bluescarni/mppp/blob/master/cmake/FindQuadmath.cmake

# Originally copied from the KDE project repository:
# http://websvn.kde.org/trunk/KDE/kdeutils/cmake/modules/FindGMP.cmake?view=markup&pathrev=675218

# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
# Copyright (c) 2008-2018 Francesco Biscani, <bluescarni@gmail.com>

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

if(Quadmath_INCLUDE_DIR AND Quadmath_LIBRARY)
	# Already in cache, be silent
	set(Quadmath_FIND_QUIETLY TRUE)
endif()

find_path(Quadmath_INCLUDE_DIR NAMES quadmath.h)
find_library(Quadmath_LIBRARY NAMES quadmath)

if(NOT Quadmath_INCLUDE_DIR OR NOT Quadmath_LIBRARY)
    cmake_push_check_state(RESET)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "quadmath")
    CHECK_CXX_SOURCE_COMPILES("
        #include <quadmath.h>

        int main(void){
            __float128 foo = ::sqrtq(123.456);
            (void) foo;
        }"
    Quadmath_USE_DIRECTLY)
    cmake_pop_check_state()
    if (Quadmath_USE_DIRECTLY)
        set(Quadmath_INCLUDE_DIR "unused" CACHE PATH "" FORCE)
        set(Quadmath_LIBRARY "quadmath" CACHE FILEPATH "" FORCE)
    endif()
endif()

find_package_handle_standard_args(Quadmath DEFAULT_MSG Quadmath_LIBRARY Quadmath_INCLUDE_DIR)

mark_as_advanced(Quadmath_INCLUDE_DIR Quadmath_LIBRARY)

# NOTE: this has been adapted from CMake's FindPNG.cmake.
if(Quadmath_FOUND AND NOT TARGET Quadmath::quadmath)
    message(STATUS "Creating the 'Quadmath::quadmath' imported target.")
    if(Quadmath_USE_DIRECTLY)
        message(STATUS "libquadmath will be included and linked directly.")
        # If we are using it directly, we must define an interface library,
        # as we do not have the full path to the shared library.
        add_library(Quadmath::quadmath INTERFACE IMPORTED)
        set_target_properties(Quadmath::quadmath PROPERTIES INTERFACE_LINK_LIBRARIES "${Quadmath_LIBRARY}")
    else()
        # Otherwise, we proceed as usual.
        add_library(Quadmath::quadmath UNKNOWN IMPORTED)
        set_target_properties(Quadmath::quadmath PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Quadmath_INCLUDE_DIR}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${Quadmath_LIBRARY}")
    endif()
endif()
