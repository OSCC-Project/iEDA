/**
 \file      attribute.h
 \brief     attribute definition for gcc/clang for extended checking
 \author    Thorsten Koch
 \version   1.0
 \date      30May2020
 \copyright Copyright (C) 2020 by Thorsten Koch <koch@zib.de>,
            licened under LGPL version 3 or later
 \details

 In case the code is compiled with gcc or clang here are defines for
 attribute which allow the compiler to better issue compile time
 warning (-Wnonnull), runtime diagnostics -fsanitize=nonnull-attribute
 and possible optimize, __attribute__ ((hot)).

 See https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes
*/
#ifndef ATTRIBUTE_H_
#define ATTRIBUTE_H_

// 755 global macro not referenced
//lint -esym(755, is_UNUSED, is_NORETURN, is_CONST, is_PURE, returns_NONNULL)
//lint -esym(755, is_MALLOC, always_INLINE, has_SIZE1, has_SIZE1x2, is_COLD, is_HOT)
//lint -esym(755, expects_NONNULL*)


#if (defined(__GNUC__) || defined(__clang__) || defined(_lint)) && !defined(__INTEL_COMPILER)

#   if defined(__clang__)
#      pragma clang diagnostic ignored "-Wignored-attributes"
#   endif // __clang__

#   define is_UNUSED         __attribute__((__unused__))            //lint !e755
#   define is_NORETURN       __attribute__((__noreturn__))          //lint !e755
#   define is_CONST          __attribute__((__const__))             //lint !e755
#   define is_PURE           __attribute__((__pure__))              //lint !e755

#   if (defined(__GNUC__) && __GNUC__ >= 8) || (defined(__clang__) && __clang_major__ >= 7 && !defined(__apple_build_version__)) || defined(_lint)
#      define fall_THROUGH __attribute__((__fallthrough__)); /*lint -fallthrough !e612 */
#      define returns_NONNULL   __attribute__((__returns_nonnull__))   //lint !e755
#   else
#      define fall_THROUGH /* FALLTHROUGH */
#      define returns_NONNULL   /**/
#   endif /* __GNUC__ >= 8 */

#else // __GNUC__ || __clang__ || _lint && !__INTEL_COMPILER

#   define is_UNUSED         /**/
#   define is_NORETURN       /**/
#   define is_CONST          /**/
#   define is_PURE           /**/
#   define returns_NONNULL   /**/
#   define fall_THROUGH       /**/

#endif // __GNUC__ || __clang__ || _lint && !__INTEL_COMPILER

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_lint) && !defined(__INTEL_COMPILER)

#   define is_MALLOC         __attribute__((__malloc__))            //lint !e755
#   define always_INLINE     __attribute__((__always_inline__))     //lint !e755
#   define has_SIZE1         __attribute__((__alloc_size__ (1)))    //lint !e755
#   define has_SIZE1x2       __attribute__((__alloc_size__ (1,2)))  //lint !e755
#   define is_COLD           __attribute__((__cold__))              //lint !e755
#   define is_HOT            __attribute__((__hot__))               //lint !e755
#   define expects_NONNULL   __attribute__((__nonnull__))           //lint !e755
#   define expects_NONNULL1  __attribute__((__nonnull__ (1)))       //lint !e755
#   define expects_NONNULL12 __attribute__((__nonnull__ (1,2)))     //lint !e755
#   define expects_NONNULL13 __attribute__((__nonnull__ (1,3)))     //lint !e755

#else // __GNUC__ || __clang__ && !_lint && !__INTEL_COMPILER

#   define is_MALLOC         /**/
#   define always_INLINE     /**/
#   define has_SIZE1         /**/
#   define has_SIZE1x2       /**/
#   define is_COLD           /**/
#   define is_HOT            /**/
#   define expects_NONNULL   /**/
#   define expects_NONNULL1  /**/
#   define expects_NONNULL12 /**/
#   define expects_NONNULL13 /**/

#endif // (__GNUC__ || __clang__) && !_lint && !__INTEL_COMPILER

#endif // ATTRIBUTE_H_
/*
 * This code is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
