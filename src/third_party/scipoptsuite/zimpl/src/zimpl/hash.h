/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: hash.h                                                        */
/*   Name....: Hash Functions                                                */
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
#ifndef _HASH_H_
#define _HASH_H_

#ifndef _NUMB_H_
#error "Need to include numb.h before hash.h"
#endif
#ifndef _ELEM_H_
#error "Need to include elem.h before hash.h"
#endif
#ifndef _TUPLE_H_
#error "Need to include tuple.h before hash.h"
#endif
#ifndef _MME_H_
#error "Need to include mme.h before hash.h (Entry,Mono)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum hash_type       { HASH_ERR = 0, HASH_TUPLE, HASH_ENTRY, HASH_ELEM_IDX, HASH_NUMB, HASH_MONO };

typedef enum hash_type           HashType;
typedef struct hash              Hash;

//lint -sem(        hash_new, chneg(2), @P >= malloc(1)) 
extern Hash*        hash_new(HashType type, int size) returns_NONNULL;
//lint -sem(        hash_free, custodial(1), inout(1), 1p == 1) 
extern void         hash_free(Hash* hash) expects_NONNULL;
//lint -sem(        hash_add_tuple, inout(1), 1p == 1, 2p == 1) 
extern void         hash_add_tuple(Hash* hash, Tuple const* tuple) expects_NONNULL;
//lint -sem(        hash_add_entry, inout(1), 1p == 1, 2p == 1) 
extern void         hash_add_entry(Hash* hash, Entry const* entry) expects_NONNULL;
//lint -sem(        hash_add_mono, inout(1), 1p == 1, 2p == 1) 
extern void         hash_add_mono(Hash* hash, Mono const* mono) expects_NONNULL;
//lint -sem(        hash_add_elem_idx, inout(1), 1p == 1, 2p == 1, chneg(3)) 
extern void         hash_add_elem_idx(Hash* hash, Elem const* elem, int idx) expects_NONNULL;
//lint -sem(        hash_add_numb, inout(1), 1p == 1, 2p == 1) 
extern void         hash_add_numb(Hash* hash, Numb const* numb) expects_NONNULL;
//lint -sem(        hash_has_tuple, 1p == 1, 2p == 1) 
extern bool         hash_has_tuple(Hash const* hash, Tuple const* tuple) expects_NONNULL;
//lint -sem(        hash_has_entry, 1p == 1, 2p == 1) 
extern bool         hash_has_entry(Hash const* hash, Tuple const* tuple) expects_NONNULL;
//lint -sem(        hash_has_numb, 1p == 1, 2p == 1) 
extern bool         hash_has_numb(Hash const* hash, Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        hash_lookup_entry, 1p == 1, 2p == 1) 
extern Entry const* hash_lookup_entry(Hash const* hash, Tuple const* tuple) expects_NONNULL;
//lint -sem(        hash_lookup_mono, 1p == 1, 2p == 1) 
extern Mono const*  hash_lookup_mono(Hash const* hash, Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        hash_lookup_elem_idx, 1p == 1, 2p == 1) 
extern int          hash_lookup_elem_idx(Hash const* hash, Elem const* elem) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _HASH_H_ 
