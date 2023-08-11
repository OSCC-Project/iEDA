/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: elem.h                                                        */
/*   Name....: Element Functions                                             */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
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
#ifndef _ELEM_H_
#define _ELEM_H_

#ifndef _NUMB_H_
#error "Need to include numb.h before elem.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum element_type
{
   ELEM_ERR = 0, ELEM_FREE, ELEM_NUMB, ELEM_STRG, ELEM_NAME
};

typedef enum element_type        ElemType;
typedef struct element           Elem;

#define ELEM_NULL  ((Elem*)0)

extern void         elem_init(void);
extern void         elem_exit(void);
//lint -sem(        elem_new_numb, 1p == 1, @P >= malloc(1)) 
extern Elem*        elem_new_numb(Numb const* n) expects_NONNULL returns_NONNULL;
//lint -sem(        elem_new_strg, 1p, @P >= malloc(1)) 
extern Elem*        elem_new_strg(char const* s) expects_NONNULL returns_NONNULL;
//lint -sem(        elem_new_name, 1p, @P >= malloc(1)) 
extern Elem*        elem_new_name(char const* s) expects_NONNULL returns_NONNULL;
//lint -sem(        elem_free, custodial(1), inout(1), 1p == 1) 
extern void         elem_free(Elem* elem) expects_NONNULL;
//lint -sem(        elem_is_valid, pure, 1p == 1) 
extern bool         elem_is_valid(Elem const* elem) is_PURE;
//lint -sem(        elem_copy, 1p == 1, @P >= malloc(1)) 
extern Elem*        elem_copy(Elem const* elem) expects_NONNULL returns_NONNULL;
//lint -sem(        elem_cmp, 1p == 1, 2p == 1) 
extern bool         elem_cmp(Elem const* elem_a, Elem const* elem_b) expects_NONNULL;
//lint -sem(        elem_get_type, pure, 1p == 1) 
extern ElemType     elem_get_type(Elem const* elem) expects_NONNULL is_PURE;
//lint -sem(        elem_get_numb, pure, 1p == 1, @p) 
extern Numb const*  elem_get_numb(Elem const* elem) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        elem_get_strg, pure, 1p == 1, @p) 
extern char const*  elem_get_strg(Elem const* elem) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        elem_get_name, pure, 1p == 1, @p) 
extern char const*  elem_get_name(Elem const* elem) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        elem_print, 1p == 1, 2p == 1) 
extern void         elem_print(FILE* fp, Elem const* elem, bool use_quotes) expects_NONNULL;
//lint -sem(        elem_hash, pure, 1p == 1) 
extern unsigned int elem_hash(Elem const* elem) expects_NONNULL is_PURE;
//lint -sem(        elem_tostr, 1p == 1, @P >= malloc(1)) 
extern char*        elem_tostr(Elem const* elem) expects_NONNULL returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _ELEM_H_ 
