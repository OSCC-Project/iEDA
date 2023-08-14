/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: inst.h                                                        */
/*   Name....: Instruction Functions                                         */
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

#ifndef _INST_H_
#define _INST_H_

#ifdef __cplusplus
extern "C" {
#endif

#define INST_NULL ((Inst)0)

/* ??? */
//lint -sem(     i_expr_sin, inout(1), 1p == 1, @P == 1P)
extern CodeNode* i_expr_sin(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_cos, inout(1), 1p == 1, @P == 1P)
extern CodeNode* i_expr_cos(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_tan, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_tan(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_asin, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_asin(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_acos, inout(1), 1p == 1, @P == 1P)
extern CodeNode* i_expr_acos(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_atan, inout(1), 1p == 1, @P == 1P)
extern CodeNode* i_expr_atan(CodeNode* self) expects_NONNULL returns_NONNULL;
   
/* inst.c
 */
//lint -sem(     i_bool_and, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_and(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_eq, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_eq(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_exists, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_exists(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_false, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_false(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_ge, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_ge(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_gt, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_gt(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_is_elem, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_is_elem(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_le, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_le(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_lt, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_lt(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_ne, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_ne(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_not, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_not(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_or, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_or(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_seq, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_seq(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_sneq, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_sneq(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_sseq, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_sseq(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_subs, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_subs(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_true, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_true(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bool_xor, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bool_xor(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_bound_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_bound_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_check, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_check(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_constraint_list, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_constraint_list(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_constraint, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_constraint(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_rangeconst, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_rangeconst(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_elem_list_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_elem_list_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_elem_list_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_elem_list_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_entry, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_entry(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_entry_list_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_entry_list_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_entry_list_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_entry_list_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_entry_list_powerset, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_entry_list_powerset(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_entry_list_subsets, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_entry_list_subsets(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_abs, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_abs(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sgn, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sgn(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_card, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_card(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_ceil, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_ceil(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_div, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_div(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_exp, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_exp(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sqrt, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sqrt(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_fac, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_fac(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_floor, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_floor(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_if_else, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_if_else(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_intdiv, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_intdiv(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_length, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_length(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_ln, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_ln(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_log, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_log(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_ord, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_ord(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_prod, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_prod(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_rand, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_rand(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_round, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_round(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sum, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sum(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_max, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_max(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_max2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_max2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sglmax, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sglmax(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_min, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_min(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_min2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_min2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sglmin, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sglmin(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_mul, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_mul(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_mod, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_mod(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_neg, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_neg(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_pow, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_pow(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_sub, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_sub(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_expr_substr, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_expr_substr(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_forall, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_forall(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_idxset_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_idxset_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_idxset_pseudo_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_idxset_pseudo_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_list_matrix, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_list_matrix(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_local_deref, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_local_deref(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_matrix_list_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_matrix_list_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_matrix_list_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_matrix_list_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newdef, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newdef(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newsym_para1, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newsym_para1(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newsym_para2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newsym_para2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newsym_set1, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newsym_set1(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newsym_set2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newsym_set2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_newsym_var, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_newsym_var(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_nop, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_nop(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_object_max, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_object_max(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_object_min, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_object_min(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_print, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_print(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_argmax, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_argmax(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_argmin, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_argmin(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_cross, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_cross(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_empty, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_empty(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_expr, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_expr(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_idxset, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_idxset(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_indexset, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_indexset(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_inter, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_inter(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_inter2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_inter2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_minus, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_minus(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_new_tuple, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_new_tuple(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_new_elem, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_new_elem(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_proj, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_proj(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_pseudo, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_pseudo(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_range, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_range(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_range2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_range2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_sdiff, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_sdiff(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_union, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_union(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_set_union2, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_set_union2(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_sos, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_sos(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_soset, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_soset(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_subto, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_subto(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_symbol_deref, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_symbol_deref(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_define_deref, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_define_deref(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_coeff, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_coeff(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_const, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_const(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_expr, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_expr(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_mul, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_mul(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_quadratic, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_power(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_power, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_sub(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_term_sum, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_term_sum(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_tuple_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_tuple_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_tuple_empty, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_tuple_empty(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_tuple_list_add, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_tuple_list_add(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_tuple_list_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_tuple_list_new(CodeNode* self) expects_NONNULL returns_NONNULL;


/* iread.c
 */
//lint -sem(     i_read_new, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_new(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read_param, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_param(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read_comment, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_comment(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read_match, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_match(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read_use, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_use(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read_skip, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read_skip(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_read, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_read(CodeNode* self) expects_NONNULL returns_NONNULL;

/* vinst.c
 */
//lint -sem(     i_vabs, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vabs(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_and, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_and(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_eq, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_eq(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_ne, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_ne(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_ge, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_ge(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_gt, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_gt(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_le, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_le(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_lt, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_lt(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_not, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_not(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_or, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_or(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vbool_xor, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vbool_xor(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vexpr_fun, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vexpr_fun(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vif, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vif(CodeNode* self) expects_NONNULL returns_NONNULL;
//lint -sem(     i_vif_else, inout(1), 1p == 1, @P == 1P) 
extern CodeNode* i_vif_else(CodeNode* self) expects_NONNULL returns_NONNULL;

//lint -sem(     addcon_as_qubo, 1p == 1 && 3p == 1 && 4p == 1) 
extern void      addcon_as_qubo(CodeNode const* self, ConType contype, Numb const* rhs,
   Term const* term_org, unsigned int flags) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _INST_H_ 
