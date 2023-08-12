/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: term.h                                                        */
/*   Name....: Term Functions                                                */
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
#ifndef _TERM_H_
#define _TERM_H_

#ifndef _MONO_H_
#error "Need to include mono.h before term.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct term              Term;

/* term.c
 */
//lint -sem(        term_new, 1n > 0, @P >= malloc(1)) 
extern Term*        term_new(int size) returns_NONNULL;
//lint -sem(        term_add_elem, inout(1), 1p == 1, 2p == 1, 3p == 1) 
extern void         term_add_elem(Term* term, Entry const* entry, Numb const* coeff, MFun mfun) expects_NONNULL;
#if 0 // ??? not used 
//lint -sem(        term_mul_elem, inout(1), 1p == 1, 2p == 1, 3p == 1) 
extern void         term_mul_elem(Term* term, Entry const* entry, Numb const* coeff) expects_NONNULL;
#endif
//lint -sem(        term_free, custodial(1), inout(1), 1p == 1) 
extern void         term_free(Term* term) expects_NONNULL;
//lint -sem(        term_is_valid, 1p == 1) 
extern bool         term_is_valid(Term const* term) is_PURE;
//lint -sem(        term_copy, 1p == 1, @P >= malloc(1)) 
extern Term*        term_copy(Term const* term) expects_NONNULL returns_NONNULL;
//lint -sem(        term_print, inout(1), 1p == 1, 2p == 1) 
extern void         term_print(FILE* fp, Term const* term, bool print_symbol_index) expects_NONNULL;
//lint -sem(        term_append_elem, inout(1), 1p == 1, inout(2), custodial(2), 2p) 
extern void         term_append_elem(Term* term, Mono* mono) expects_NONNULL;
//lint -sem(        term_append_term, inout(1), 1p == 1, 2p == 1) 
extern void         term_append_term(Term* term_a, Term const* term_b) expects_NONNULL;
//lint -sem(        term_add_term, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Term*        term_add_term(Term const* term_a, Term const* term_b) expects_NONNULL returns_NONNULL;
//lint -sem(        term_sub_term, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Term*        term_sub_term(Term const* term_a, Term const* term_b) expects_NONNULL returns_NONNULL;
//lint -sem(        term_mul_term, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Term*        term_mul_term(Term const* term_a, Term const* term_b) expects_NONNULL returns_NONNULL;
//lint -sem(        term_simplify, 1p, @P >= malloc(1)) 
extern Term*        term_simplify(Term const* term_org) expects_NONNULL returns_NONNULL;
//lint -sem(        term_add_constant, inout(1), 1p == 1, 2p == 1) 
extern void         term_add_constant(Term* term, Numb const* value) expects_NONNULL;
//lint -sem(        term_sub_constant, inout(1), 1p == 1, 2p == 1) 
extern void         term_sub_constant(Term* term, Numb const* value) expects_NONNULL;
//lint -sem(        term_mul_coeff, inout(1), 1p == 1, 2p == 1) 
extern void         term_mul_coeff(Term* term, Numb const* value) expects_NONNULL;
//lint -sem(        term_get_constant, 1p == 1) 
extern Numb const*  term_get_constant(Term const* term) expects_NONNULL returns_NONNULL;
#if 0 // ??? not used 
//lint -sem(        term_negate, inout(1), 1p == 1) 
extern void         term_negate(Term* term) expects_NONNULL;
#endif
//lint -sem(        term_to_objective, 1p == 1) 
extern void         term_to_objective(Term const* term) expects_NONNULL;
//lint -sem(        term_get_elements, 1p == 1, chneg(@)) 
extern int          term_get_elements(Term const* term) expects_NONNULL is_PURE;
//lint -sem(        term_get_element, 1p == 1, chneg(2), @p == 1) 
extern Mono*        term_get_element(Term const* term, int i) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        term_get_lower_bound, 1p == 1, @P >= malloc(1)) 
extern Bound*       term_get_lower_bound(Term const* term) expects_NONNULL returns_NONNULL;
//lint -sem(        term_get_upper_bound, 1p == 1, @P >= malloc(1)) 
extern Bound*       term_get_upper_bound(Term const* term) expects_NONNULL returns_NONNULL;
//lint -sem(        term_is_all_integer, 1p == 1) 
extern bool         term_is_all_integer(Term const* term) expects_NONNULL;
//lint -sem(        term_is_linear, 1p == 1) 
extern bool         term_is_linear(Term const* term) expects_NONNULL is_PURE;
//lint -sem(        term_is_polynomial, 1p == 1) 
extern bool         term_is_polynomial(Term const* term) expects_NONNULL is_PURE;
//lint -sem(        term_has_realfunction, 1p == 1) 
extern bool         term_has_realfunction(Term const* term) expects_NONNULL is_PURE;
//lint -sem(        term_get_degree, 1p == 1) 
extern int          term_get_degree(Term const* term) expects_NONNULL is_PURE;
//lint -sem(        term_make_conditional, 1p == 1, 2p == 1, @P > malloc(1P)) 
extern Term*        term_make_conditional(Term const* ind_term, Term const* cond_term, bool is_true) expects_NONNULL returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _TERM_H_ 
