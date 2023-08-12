/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: set4.c                                                        */
/*   Name....: Set Functions                                                 */
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
#ifndef _SET_H_
#define _SET_H_

#ifndef _MME_H_
#error "Need to include mme.h before set.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum set_check_type
{
   SET_CHECK_NONE, SET_CHECK_QUIET, SET_CHECK_WARN
};

typedef enum set_check_type      SetCheckType;

typedef long long                SetIterIdx;

typedef union set                Set;
typedef union set_iter           SetIter;

extern void         set_init(void);
extern void         set_exit(void);
//lint -sem(        set_free, custodial(1), inout(1), 1p == 1) 
extern void         set_free(Set* set) expects_NONNULL;
//lint -sem(        set_is_valid, 1p == 1, pure) 
extern bool         set_is_valid(Set const* set) is_PURE;
//lint -sem(        set_copy, 1p == 1, @P >= malloc(1)) 
extern Set*         set_copy(Set const* set) expects_NONNULL returns_NONNULL;
//lint -sem(        set_lookup, 1p == 1, 2p == 1) 
extern bool         set_lookup(Set const* set, Tuple const* tuple) expects_NONNULL;
//lint -sem(        set_iter_init, 1p == 1, @P >= malloc(1)) 
extern SetIter*     set_iter_init(Set const* set, Tuple const* pattern) expects_NONNULL1 returns_NONNULL;
//lint -sem(        set_iter_next, inout(1), 1p == 1, 2p == 1, r_null) 
extern Tuple*       set_iter_next(SetIter* iter, Set const* set) expects_NONNULL;
//lint -sem(        set_iter_exit, custodial(1), inout(1), 1p == 1, 2p == 1) 
extern void         set_iter_exit(SetIter* iter, Set const* set) expects_NONNULL;
//lint -sem(        set_get_dim, 1p == 1, chneg(@)) 
extern int          set_get_dim(Set const* set) expects_NONNULL is_PURE;
//lint -sem(        set_get_members, 1p == 1, chneg(@)) 
extern int          set_get_members(Set const* set) expects_NONNULL is_PURE;
//lint -sem(        set_get_tuple, 1p == 1, chneg(2), @P >= 1) 
extern Tuple*       set_get_tuple(Set const* set, SetIterIdx idx) expects_NONNULL returns_NONNULL;
//lint -sem(        set_print, 1p == 1, 2p == 1) 
extern void         set_print(FILE* fp, Set const* set) expects_NONNULL;

//lint -sem(        set_empty_new, chneg(1), @P >= malloc(1)) 
extern Set*         set_empty_new(int dim) returns_NONNULL;

//lint -sem(        set_pseudo_new, @P >= malloc(1)) 
extern Set*         set_pseudo_new(void) returns_NONNULL;
//lint -sem(        set_new_from_list, 1p == 1, @P >= malloc(1)) 
extern Set*         set_new_from_list(List const* list, SetCheckType check) expects_NONNULL returns_NONNULL;
//lint -sem(        set_range_new, @P >= malloc(1)) 
extern Set*         set_range_new(int begin, int end, int step) returns_NONNULL;
//lint -sem(        set_prod_new, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_prod_new(Set const* a, Set const* b) expects_NONNULL returns_NONNULL;
//lint -sem(        set_union, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_union(Set const* seta, Set const* setb) expects_NONNULL returns_NONNULL;
//lint -sem(        set_inter, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_inter(Set const* set_a, Set const* set_b) expects_NONNULL returns_NONNULL;
//lint -sem(        set_minus, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_minus(Set const* set_a, Set const* set_b) expects_NONNULL returns_NONNULL;
//lint -sem(        set_sdiff, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_sdiff(Set const* set_a, Set const* set_b) expects_NONNULL returns_NONNULL;
//lint -sem(        set_proj, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Set*         set_proj(Set const* set_a, Tuple const* pattern) expects_NONNULL returns_NONNULL;
//lint -sem(        set_is_subseteq, 1p == 1, 2p == 1) 
extern bool         set_is_subseteq(Set const* set_a, Set const* set_b) expects_NONNULL;
//lint -sem(        set_is_subset, 1p == 1, 2p == 1) 
extern bool         set_is_subset(Set const* set_a, Set const* set_b) expects_NONNULL;
//lint -sem(        set_is_equal, 1p == 1, 2p == 1) 
extern bool         set_is_equal(Set const* set_a, Set const* set_b) expects_NONNULL;
//lint -sem(        set_subset_list, 1p == 1, 2n > 0, 4p == 1, @P > malloc(1)) 
extern List*        set_subsets_list(
   Set const* set, int subset_size, List* list, SetIterIdx* idx) expects_NONNULL1 returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _SET_H_ 
