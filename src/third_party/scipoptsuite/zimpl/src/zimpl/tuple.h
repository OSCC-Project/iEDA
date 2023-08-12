/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: tuple.h                                                       */
/*   Name....: Tuple Functions                                               */
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
#ifndef _TUPLE_H_
#define _TUPLE_H_

#ifndef _ELEM_H_
#error "Need to include elem.h before tuple.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tuple Tuple;

#define TUPLE_NULL ((Tuple*)0)

//lint -sem(        tuple_new, chneg(1), @P >= malloc(1)) 
extern Tuple*       tuple_new(int dim) returns_NONNULL;
//lint -sem(        tuple_free, custodial(1), inout(1), 1p == 1) 
extern void         tuple_free(Tuple* tuple) expects_NONNULL;
//lint -sem(        tuple_is_valid, pure, 1p == 1) 
extern bool         tuple_is_valid(Tuple const* tuple) is_PURE;
//lint -sem(        tuple_copy, 1p == 1, @P >= malloc(1)) 
extern Tuple*       tuple_copy(Tuple const* tuple) expects_NONNULL returns_NONNULL;
//lint -sem(        tuple_cmp, 1p == 1 && 2p == 1) 
   extern bool         tuple_cmp(Tuple const* tuple_a, Tuple const* tuple_b) expects_NONNULL;
//lint -sem(        tuple_get_dim, pure, 1p == 1, chneg(@)) 
extern int          tuple_get_dim(Tuple const* tuple) expects_NONNULL is_PURE;
//lint -sem(        tuple_set_elem, custodial(3), 1p == 1, chneg(2), 3p == 1) 
extern void         tuple_set_elem(Tuple* tuple, int idx, Elem* elem) expects_NONNULL;
//lint -sem(        tuple_get_elem, 1p == 1, chneg(2), @p == 1) 
extern Elem const*  tuple_get_elem(Tuple const* tuple, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        tuple_combine, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Tuple*       tuple_combine(Tuple const* ta, Tuple const* tb) expects_NONNULL returns_NONNULL;
//lint -sem(        tuple_print, 1p == 1, 2p == 1) 
extern void         tuple_print(FILE* fp, Tuple const* tuple) expects_NONNULL;
//lint -sem(        tuple_hash, pure, 1p == 1) 
extern unsigned int tuple_hash(Tuple const* tuple) expects_NONNULL is_PURE;
//lint -sem(        tuple_tostr, 1p == 1, @P >= malloc(1)) 
extern char*        tuple_tostr(Tuple const* tuple) expects_NONNULL returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _TUPLE_H_ 

