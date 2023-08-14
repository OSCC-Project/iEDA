/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: symbol.h                                                      */
/*   Name....: Symbol Table Functions                                        */
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
#ifndef _SYMBOL_H_
#define _SYMBOL_H_

#ifndef _TUPLE_H_
#error "Need to include tuple.h before symbol.h"
#endif
#ifndef _SET_H_
#error "Need to include set.h before symbol.h"
#endif
#ifndef _MME_H_
#error "Need to include mme.h before symbol.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(        symbol_new, 1p, 3p == 1, chneg(4), @p == 1) 
extern Symbol*      symbol_new(char const* name,
   SymbolType type, Set const* set, int estimated_size, Entry const* deflt) expects_NONNULL13 returns_NONNULL;
extern void         symbol_exit(void);
//lint -sem(        symbol_is_valid, 1p == 1) 
extern bool         symbol_is_valid(Symbol const* symbol) is_PURE;
//lint -sem(        symbol_lookup, 1p, r_null) 
extern Symbol*      symbol_lookup(char const* name) expects_NONNULL;
//lint -sem(        symbol_has_entry, 1p == 1, 2p == 1) 
extern bool         symbol_has_entry(Symbol const* sym, Tuple const* tuple) expects_NONNULL;
//lint -sem(        symbol_lookup_entry, 1p == 1, 2p == 1) 
extern Entry const* symbol_lookup_entry(Symbol const* sym, Tuple const* tuple) expects_NONNULL;
//lint -sem(        symbol_add_entry, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         symbol_add_entry(Symbol* sym, Entry* entry) expects_NONNULL;
//lint -sem(        symbol_get_dim, 1p == 1, chneg(@)) 
extern int          symbol_get_dim(Symbol const* sym) expects_NONNULL is_PURE;
//lint -sem(        symbol_get_iset, 1p == 1, @p == 1) 
extern Set const*   symbol_get_iset(Symbol const* sym) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_get_name, 1p == 1, @p) 
extern char const*  symbol_get_name(Symbol const* sym) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_get_type, 1p == 1) 
extern SymbolType   symbol_get_type(Symbol const* sym) expects_NONNULL is_PURE;
//lint -sem(        symbol_get_numb, 1p == 1, chneg(2), @p == 1) 
extern Numb const*  symbol_get_numb(Symbol const* sym, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_get_strg, 1p == 1, chneg(2), @p) 
extern char const*  symbol_get_strg(Symbol const* sym, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_get_set, 1p == 1, chneg(2), @p == 1) 
extern Set const*   symbol_get_set(Symbol const* sym, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_get_var, 1p == 1, chneg(2), @p == 1) 
extern Var*         symbol_get_var(Symbol const* sym, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        symbol_print, inout(1), 1p == 1, 2p == 1) 
extern void         symbol_print(FILE* fp, Symbol const* sym) expects_NONNULL;
//lint -sem(        symbol_print_all, inout(1), 1p == 1) 
extern void         symbol_print_all(FILE* fp) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _SYMBOL_H_ 
