/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: numb.c                                                        */
/*   Name....: Number Functions                                              */
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
#ifndef _NUMB_H_
#define _NUMB_H_

typedef struct number            Numb;

 /* numbgmp.c
 */
extern void         numb_init(bool with_management);
extern void         numb_exit(void);
//lint -sem(        numb_new, @P >= malloc(1)) 
extern Numb*        numb_new(void) returns_NONNULL;
//lint -sem(        numb_new_ascii, 1p, @P >= malloc(1)) 
extern Numb*        numb_new_ascii(char const* val) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_new_integer, @P >= malloc(1)) 
extern Numb*        numb_new_integer(int val) returns_NONNULL;
//lint -sem(        numb_new_longlong, @P >= malloc(1)) 
extern Numb*        numb_new_longlong(long long val) returns_NONNULL;

//lint -sem(        numb_free, custodial(1), inout(1), 1p == 1) 
extern void         numb_free(Numb* numb) expects_NONNULL;
//lint -sem(        numb_is_valid, pure, 1p == 1) 
extern bool         numb_is_valid(Numb const* numb) is_PURE;

//lint -sem(        numb_copy, 1p == 1, @P >= malloc(1)) 
extern Numb*        numb_copy(Numb const* source) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_equal, pure, 1p == 1, 2p == 1) 
extern bool         numb_equal(Numb const* numb_a, Numb const* numb_b) expects_NONNULL is_PURE;
//lint -sem(        numb_cmp, pure, 1p == 1, 2p == 1) 
extern int          numb_cmp(Numb const* numb_a, Numb const* numb_b) expects_NONNULL is_PURE;
//lint -sem(        numb_set, inout(1), 1p == 1, 2p == 1) 
extern void         numb_set(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_add, inout(1), 1p == 1, 2p == 1) 
extern void         numb_add(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_add, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_add(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_sub, inout(1), 1p == 1, 2p == 1) 
extern void         numb_sub(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_sub, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_sub(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_mul, inout(1), 1p == 1, 2p == 1) 
extern void         numb_mul(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_mul, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_mul(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_div, inout(1), 1p == 1, 2p == 1) 
extern void         numb_div(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_div, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_div(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_intdiv, inout(1), 1p == 1, 2p == 1) 
extern void         numb_intdiv(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_intdiv, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_intdiv(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_new_pow, 1p == 1, chneg(2), @P >= malloc(1)) 
extern Numb*        numb_new_pow(Numb const* base, int expo) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_new_fac, chneg(1), @P >= malloc(1)) 
extern Numb*        numb_new_fac(int n) returns_NONNULL;
//lint -sem(        numb_mod, inout(1), 1p == 1, 2p == 1) 
extern void         numb_mod(Numb* numb_a, Numb const* numb_b) expects_NONNULL;
//lint -sem(        numb_new_mod, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_mod(Numb const* numb_a, Numb const* numb_b) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_neg, inout(1), 1p == 1) 
extern void         numb_neg(Numb* numb) expects_NONNULL;
//lint -sem(        numb_abs, inout(1), 1p == 1) 
extern void         numb_abs(Numb* numb) expects_NONNULL;
//lint -sem(        numb_sgn, inout(1), 1p == 1) 
extern void         numb_sgn(Numb* numb) expects_NONNULL;
//lint -sem(        numb_get_sgn, 1p == 1, @n >= -1 && @n <= 1) 
extern int          numb_get_sgn(Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        numb_round, inout(1), 1p == 1) 
extern void         numb_round(Numb* numb) expects_NONNULL;
//lint -sem(        numb_ceil, inout(1), 1p == 1) 
extern void         numb_ceil(Numb* numb) expects_NONNULL;
//lint -sem(        numb_floor, inout(1), 1p == 1) 
extern void         numb_floor(Numb* numb) expects_NONNULL;
//lint -sem(        numb_new_log, 1p == 1, @P >= malloc(1) || @P == 0) 
extern Numb*        numb_new_log(Numb const* numb) expects_NONNULL;
//lint -sem(        numb_new_sqrt, 1p == 1, @P >= malloc(1) || @P == 0) 
extern Numb*        numb_new_sqrt(Numb const* numb) expects_NONNULL;
//lint -sem(        numb_new_exp, 1p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_exp(Numb const* numb) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_new_ln, 1p == 1, @P >= malloc(1) || @P == 0) 
extern Numb*        numb_new_ln(Numb const* numb) expects_NONNULL;
//lint -sem(        numb_new_rand, 1p == 1, 2p == 1, @P >= malloc(1)) 
extern Numb*        numb_new_rand(Numb const* mini, Numb const* maxi) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_todbl, pure, 1p == 1) 
extern double       numb_todbl(Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        numb_print, inout(1), 1p == 1, 2p == 1) 
extern void         numb_print(FILE* fp, Numb const* numb) expects_NONNULL;
//lint -sem(        numb_hash, pure, 1p == 1) 
extern unsigned int numb_hash(Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        numb_tostr, 1p == 1, @p) 
extern char*        numb_tostr(Numb const* numb) expects_NONNULL returns_NONNULL;
//lint -sem(        numb_zero, pure, @p == 1) 
extern Numb const*  numb_zero(void) returns_NONNULL is_CONST;
//lint -sem(        numb_one, pure, @p == 1) 
extern Numb const*  numb_one(void) returns_NONNULL is_CONST;
//lint -sem(        numb_minusone, pure, @p == 1) 
extern Numb const*  numb_minusone(void) returns_NONNULL is_CONST;
//lint -sem(        numb_unknown, pure, @p == 1) 
extern Numb const*  numb_unknown(void) returns_NONNULL is_CONST;
//lint -sem(        numb_is_int, pure, 1p == 1) 
extern bool         numb_is_int(Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        numb_toint, pure, 1p == 1) 
extern int          numb_toint(Numb const* numb) expects_NONNULL is_PURE;
//lint -sem(        numb_is_number, 1p == 1) 
extern bool         numb_is_number(const char *s) expects_NONNULL is_PURE;

#ifdef __GMP_H__
//lint -sem(        numb_new_mpq, @P >= malloc(1)) 
extern Numb*        numb_new_mpq(const mpq_t val) returns_NONNULL;
//lint -sem(        numb_new_mpq, 1p == 1) 
extern void         numb_get_mpq(Numb const* numb, mpq_t value) expects_NONNULL;
#endif // __GMP_H__ 

#endif // _NUMB_H_ 
