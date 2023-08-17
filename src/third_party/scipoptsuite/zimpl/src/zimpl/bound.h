/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: bound.h                                                       */
/*   Name....: Bound value                                                   */
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
#ifndef _BOUND_H_
#define _BOUND_H_

#ifndef _NUMB_H_
#error "Need to include numb.h before bound.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum bound_type
{
   BOUND_ERROR = 0, BOUND_VALUE, BOUND_INFTY, BOUND_MINUS_INFTY
};

typedef enum bound_type BoundType;
typedef struct bound    Bound;

//lint -sem(        bound_new, @P >= malloc(1)) 
extern Bound*       bound_new(BoundType type, Numb const* value) returns_NONNULL;
//lint -sem(        bound_free, custodial(1), inout(1), 1p == 1) 
extern void         bound_free(Bound* bound) expects_NONNULL;
//lint -sem(        bound_is_valid, 1p == 1) 
extern bool         bound_is_valid(Bound const* bound) is_PURE;
//lint -sem(        bound_copy, 1p == 1, @P >= malloc(1)) 
extern Bound*       bound_copy(Bound const* source) expects_NONNULL returns_NONNULL;
//lint -sem(        bound_get_type, 1p == 1) 
extern BoundType    bound_get_type(Bound const* bound) expects_NONNULL is_PURE;
//lint -sem(        bound_get_value, 1p == 1, @p == 1) 
extern Numb const*  bound_get_value(Bound const* bound) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        bound_print, inout(1), 1p == 1, 2p == 1) 
extern void         bound_print(FILE* fp, Bound const* bound) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _BOUND_H_ 

