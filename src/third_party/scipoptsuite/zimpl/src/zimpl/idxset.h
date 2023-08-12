/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: idxset.h                                                      */
/*   Name....: IndexSet Functions                                            */
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
#ifndef _IDXSET_H_
#define _IDXSET_H_

#ifndef _TUPLE_H_
#error "Need to include tuple.h before idxset.h"
#endif
#ifndef _SET_H_
#error "Need to include set.h before idxset.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct index_set         IdxSet;

/* idxset.c
 */
//lint -sem(        idxset_new, 1p == 1, 2p == 1, inout(3), 3p == 1, @P >= malloc(1)) 
extern IdxSet*      idxset_new(
   Tuple const* tuple, Set const* set, CodeNode* lexpr, bool is_unrestricted) expects_NONNULL returns_NONNULL;
//lint -sem(        idxset_free, custodial(1), inout(1), 1p == 1) 
extern void         idxset_free(IdxSet* idxset) expects_NONNULL;
//lint -sem(        idxset_is_valid, pure, 1p == 1) 
extern bool         idxset_is_valid(IdxSet const* idxset) is_PURE;
//lint -sem(        idxset_copy, 1p == 1, @P >= malloc(1)) 
extern IdxSet*      idxset_copy(IdxSet const* source) expects_NONNULL returns_NONNULL;
//lint -sem(        idxset_get_lexpr, 1p == 1, @p == 1) 
extern CodeNode*    idxset_get_lexpr(IdxSet const* idxset) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        idxset_get_tuple, 1p == 1, @p == 1) 
extern Tuple const* idxset_get_tuple(IdxSet const* idxset) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        idxset_get_set, 1p == 1, @p == 1) 
extern Set const*   idxset_get_set(IdxSet const* idxset) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        idxset_is_unrestricted, 1p == 1) 
extern bool         idxset_is_unrestricted(IdxSet const* idxset) expects_NONNULL is_PURE;
//lint -sem(        idxset_print, 1p == 1, inout(1), 2p == 1) 
extern void         idxset_print(FILE* fp, IdxSet const* idxset) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _IDXSET_H_ 
