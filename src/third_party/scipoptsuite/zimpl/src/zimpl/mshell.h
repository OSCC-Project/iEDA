/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mshell.h                                                      */
/*   Name....: Memory Allocation Shell                                       */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2007-2022 by Thorsten Koch <koch@zib.de>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef _MSHELL_H_
#define _MSHELL_H_  

#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(  mem_malloc, 1n > 0, 2p, @P == malloc(1n)) 
extern void*  mem_malloc(size_t, char const*, const int) is_MALLOC returns_NONNULL;
//lint -sem(  mem_calloc, 1n > 0, 2n > 0, 3p, @P == malloc(1n * 2n)) 
extern void*  mem_calloc(size_t, size_t, char const*, const int) is_MALLOC returns_NONNULL;
//lint -sem(  mem_realloc, custodial(1), 1p, 2n > 0, 3p, @P == malloc(2n)) 
extern void*  mem_realloc(void*, size_t, char const*, const int) is_MALLOC returns_NONNULL;
//lint -sem(  mem_strdup, 1p > 0, 2p, @p == malloc(1p)) 
extern char*  mem_strdup(char const*, char const*, const int) is_MALLOC returns_NONNULL;
//lint -sem(  mem_free, custodial(1), 1p > 0, 2p) 
extern void   mem_free(void*, char const*, const int);

/* realloc, free, and strdup expect non NULL pointer arguments.
   But since they will check for wrong behaviour this is not conveyed to the compiler.
*/
   
#ifndef _MSHELL_C_ 

#ifdef strdup
#undef strdup
#endif // strdup 

//lint -e652 -e683
#define malloc(a)       mem_malloc((a), __FILE__, __LINE__)
#define calloc(a, b)    mem_calloc((a), (b), __FILE__, __LINE__)
#define realloc(a, b)   mem_realloc((a), (b), __FILE__, __LINE__)
#define strdup(a)       mem_strdup((a), __FILE__, __LINE__)
#define free(a)         mem_free((a), __FILE__, __LINE__) 
//lint +e652 +e683

#endif // _MSHELL_C_ 

#ifndef NO_MSHELL 

extern size_t mem_used(void) is_PURE;
extern void   mem_maximum(FILE* fp);
extern void   mem_display(FILE* fp);
extern void   mem_check_x(void const* p, char const* file, const int line);
extern void   mem_check_all_x(char const* file, const int line);
extern void   mem_hide_x(void* p, char const* file, const int line);

#define mem_check(a)    mem_check_x(a, __FILE__, __LINE__)
#define mem_check_all() mem_check_all_x(__FILE__, __LINE__)
#define mem_hide(a)     mem_hide_x(a, __FILE__, __LINE__)

#else /* NO_MSHELL */

#define mem_used()          /**/
#define mem_maximum(a)      /**/
#define mem_display(a)      /**/
#define mem_check(a)        /**/
#define mem_check_all()     /**/
#define mem_hide(a)         /**/

#endif /* !NO_MSHELL */

#ifdef __cplusplus
}
#endif

#endif /* _MSHELL_H_ */


