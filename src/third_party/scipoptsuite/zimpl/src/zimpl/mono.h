/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mono.h                                                        */
/*   Name....: Monom Functions                                               */
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
#ifndef _MONO_H_
#define _MONO_H_

#ifdef __cplusplus
extern "C" {
#endif

enum mono_function {
   MFUN_SQRT = -2, MFUN_NONE = 0,  MFUN_TRUE = 1, MFUN_FALSE = 2, MFUN_LOG = 3,
   MFUN_EXP  = 4,  MFUN_LN   = 5,  MFUN_SIN  = 6, MFUN_COS   = 7, MFUN_TAN = 8,
   MFUN_ABS  = 9,  MFUN_SGN  = 10, MFUN_POW = 11, MFUN_SGNPOW = 12
};

typedef enum   mono_function MFun;
typedef struct mono_element  MonoElem;

struct mono_element
{
   SID
   Entry*    entry;
   MonoElem* next;
};

/* if fun is != MFUN_NONE, the meaning of (fun, elem, coeff) is
 * fun(elem). coeff is not used, unless fun is MFUN_POW oder MFUN_SGNPOW
 * in which case coeff is the second argument, i.e. fun(elem, coeff).
 */
struct mono
{
   SID
   int       count; // Only needed to hopefully compare monoms faster. 
   MFun      fun;
   Numb*     coeff; 
   MonoElem  first;
};

//lint -sem(        mono_new, 1P >= 1, 2P >= 1, @P >= malloc(1)) 
extern Mono*        mono_new(Numb const* coeff, Entry const* entry, MFun fun) expects_NONNULL returns_NONNULL;
//lint -sem(        mono_is_valid, pure, 1P >= 1) 
extern bool         mono_is_valid(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_free, custodial(1), inout(1), 1P >= 1) 
extern void         mono_free(Mono* mono) expects_NONNULL;
//lint -sem(        mono_mul_entry, inout(1), 1P >= 1, 2P >= 1) 
extern void         mono_mul_entry(Mono* mono, Entry const* entry) expects_NONNULL;
//lint -sem(        mono_copy, 1P >= 1, @P == malloc(1P)) 
extern Mono*        mono_copy(Mono const* mono) expects_NONNULL returns_NONNULL;
//lint -sem(        mono_mul_coeff, inout(1), 1P >= 1, 2P >= 1) 
extern void         mono_mul_coeff(Mono const* term, Numb const* value) expects_NONNULL;
//lint -sem(        mono_add_coeff, inout(1), 1P >= 1, 2P >= 1) 
extern void         mono_add_coeff(Mono const* term, Numb const* value) expects_NONNULL;
//lint -sem(        mono_hash, pure, 1P >= 1) 
extern unsigned int mono_hash(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_equal, pure, 1P >= 1, 2P >= 1) 
extern bool         mono_equal(Mono const* ma, Mono const* mb) expects_NONNULL is_PURE;
//lint -sem(        mono_mul, 1P >= 1, 2P >= 1, @P >= malloc(1P)) 
extern Mono*        mono_mul(Mono const* ma, Mono const* mb) expects_NONNULL returns_NONNULL;
//lint -sem(        mono_neg, inout(1), 1P >= 1) 
extern void         mono_neg(Mono* mono) expects_NONNULL;
//lint -sem(        mono_is_linear, pure, 1P >= 1) 
extern bool         mono_is_linear(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_get_degree, pure, 1P >= 1, @n >= 1) 
extern int          mono_get_degree(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_get_coeff, pure, 1P >= 1, @P >= 1) 
extern Numb const*  mono_get_coeff(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_set_function, inout(1), 1P >= 1) 
extern void         mono_set_function(Mono* mono, MFun f) expects_NONNULL;
//lint -sem(        mono_get_function, pure, 1P >= 1) 
extern MFun         mono_get_function(Mono const* mono) expects_NONNULL is_PURE;
//lint -sem(        mono_get_var, 1P >= 1, chneg(2), @P >= 1) 
extern Var*         mono_get_var(Mono const* mono, int idx) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        mono_print, 1P >= 1, 2P >= 1) 
extern void         mono_print(FILE* fp, Mono const* mono, bool print_symbol_index) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _MONO_H_

