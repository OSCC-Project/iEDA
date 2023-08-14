/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: list.h                                                        */
/*   Name....: List Functions                                                */
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
#ifndef _LIST_H_
#define _LIST_H_

#ifndef _MME_H_
#error "Need to include mme.h before list.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_NULL ((List*)0)

//lint -sem(        list_new_elem, 1p == 1, @P >= malloc(1)) 
extern List*        list_new_elem(Elem const* elem) expects_NONNULL returns_NONNULL;
//lint -sem(        list_new_tuple, 1p == 1, @P >= malloc(1)) 
extern List*        list_new_tuple(Tuple const* tuple) expects_NONNULL returns_NONNULL;
//lint -sem(        list_new_entry, 1p == 1, @P >= malloc(1)) 
extern List*        list_new_entry(Entry const* entry) expects_NONNULL returns_NONNULL;
//lint -sem(        list_new_list, 1p == 1, @P >= malloc(1)) 
extern List*        list_new_list(List const* list) expects_NONNULL returns_NONNULL;
//lint -sem(        list_free, custodial(1), inout(1), 1p == 1) 
extern void         list_free(List* list) expects_NONNULL;
//lint -sem(        list_is_valid, pure, 1p == 1) 
extern bool         list_is_valid(List const* list) is_PURE;
//lint -sem(        list_is_elemlist, pure, 1p == 1) 
extern bool         list_is_elemlist(List const* list) expects_NONNULL is_PURE;
//lint -sem(        list_is_entrylist, pure, 1p == 1) 
extern bool         list_is_entrylist(List const* list) expects_NONNULL is_PURE;
//lint -sem(        list_is_tuplelist, pure, 1p == 1) 
extern bool         list_is_tuplelist(List const* list) expects_NONNULL is_PURE;
//lint -sem(        list_copy, 1p == 1, @P >= malloc(1)) 
extern List*        list_copy(List const* list) expects_NONNULL returns_NONNULL;
//lint -sem(        list_add_list, inout(1), 1p == 1, 2p == 1) 
extern void         list_add_list(List* list, List const* ll) expects_NONNULL;
//lint -sem(        list_add_elem, inout(1), 1p == 1, 2p == 1) 
extern void         list_add_elem(List* list, Elem const* elem) expects_NONNULL;
//lint -sem(        list_add_tuple, inout(1), 1p == 1, 2p == 1) 
extern void         list_add_tuple(List* list, Tuple const* tuple) expects_NONNULL;
//lint -sem(        list_add_entry, inout(1), 1p == 1, 2p == 1) 
extern void         list_add_entry(List* list, Entry const* entry) expects_NONNULL;
//lint -sem(        list_insert_elem, inout(1), 1p == 1, 2p == 1) 
extern void         list_insert_elem(List* list, Elem const* elem) expects_NONNULL;
//lint -sem(        list_insert_tuple, inout(1), 1p == 1, 2p == 1) 
extern void         list_insert_tuple(List* list, Tuple const* tuple) expects_NONNULL;
//lint -sem(        list_insert_entry, inout(1), 1p == 1, 2p == 1) 
extern void         list_insert_entry(List* list, Entry const* entry) expects_NONNULL;
//lint -sem(        list_get_elems, pure, 1p == 1, chneg(@)) 
extern int          list_get_elems(List const* list) expects_NONNULL is_PURE;
//lint -sem(        list_get_elem, 1p == 1, 2p, r_null) 
extern Elem const*  list_get_elem(List const* list, ListElem** idxp) expects_NONNULL;
//lint -sem(        list_get_tuple, 1p == 1, 2p, r_null) 
extern Tuple const* list_get_tuple(List const* list, ListElem** idxp) expects_NONNULL;
//lint -sem(        list_get_entry, 1p == 1, 2p r_null) 
extern Entry const* list_get_entry(List const* list, ListElem** idxp) expects_NONNULL;
//lint -sem(        list_get_list, 1p == 1, 2p, r_null) 
extern List const*  list_get_list(List const* list, ListElem** idxp) expects_NONNULL;
//lint -sem(        list_print, inout(1), 1p == 1, 2p == 1) 
extern void         list_print(FILE* fp, List const* list) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _LIST_H_ 
