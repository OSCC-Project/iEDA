/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: code.h                                                        */
/*   Name....: Code Functions                                                */
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

#ifndef _CODE_H_
#define _CODE_H_

#ifndef _RATLPTYPES_H_
#error "Need to include ratlptypes.h before code.h"
#endif
#ifndef _MME_H_
#error "Need to include mme.h before code.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(        code_new_inst, 1p == 1, chneg(2), @P > malloc(1)) 
extern CodeNode*    code_new_inst(Inst inst, int childs, ...) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_numb, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_numb(Numb* numb) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_strg, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_strg(char const* strg) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_name, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_name(char const* name) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_size, chneg(1), @P >= malloc(1)) 
extern CodeNode*    code_new_size(int size) returns_NONNULL;
//lint -sem(        code_new_varclass, @P >= malloc(1)) 
extern CodeNode*    code_new_varclass(VarClass varclass) returns_NONNULL;
//lint -sem(        code_new_contype, @P >= malloc(1)) 
extern CodeNode*    code_new_contype(ConType contype) returns_NONNULL;
//lint -sem(        code_new_bits, @P >= malloc(1)) 
extern CodeNode*    code_new_bits(unsigned int bits) returns_NONNULL;
//lint -sem(        code_new_symbol, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_symbol(Symbol* sym) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_define, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_define(Define* def) expects_NONNULL returns_NONNULL;
//lint -sem(        code_new_bound, custodial(1), 1p == 1, @P >= malloc(1))
extern CodeNode*    code_new_bound(BoundType type) returns_NONNULL;
//lint -sem(        code_free, custodial(1), inout(1), 1p == 1) 
extern void         code_free(CodeNode* node) expects_NONNULL;
//lint -sem(        code_free_value, custodial(1), inout(1), 1p == 1) 
extern void         code_free_value(CodeNode* node) expects_NONNULL;
//lint -sem(        code_is_valid, pure, 1p == 1) 
extern bool         code_is_valid(CodeNode const* node) is_PURE;
//lint -sem(        code_get_type, pure, 1p == 1) 
extern CodeType     code_get_type(CodeNode const* node) expects_NONNULL is_PURE;
//lint -sem(        code_get_inst, pure, 1p == 1, @p) 
extern Inst         code_get_inst(CodeNode const* node) expects_NONNULL is_PURE;
//lint -sem(        code_set_root, custodial(1), 1p == 1) 
extern void         code_set_root(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_root, pure, @p == 1) 
extern CodeNode*    code_get_root(void) returns_NONNULL;
//lint -sem(        code_set_child, 1p == 1, chneg(2), custodial(3), 3p == 1) 
extern void         code_set_child(CodeNode* node, int idx, CodeNode* child) expects_NONNULL;
//lint -sem(        code_errmsg, 1p == 1) 
extern void         code_errmsg(CodeNode const* node) expects_NONNULL;
//lint -sem(        code_eval, 1p == 1, @p == 1) 
extern CodeNode*    code_eval(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_prune_tree, inout(1), 1p == 1) 
extern bool         code_prune_tree(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_child, pure, 1p == 1, chneg(2), @p == 1) 
extern CodeNode*    code_get_child(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_numb, 1p == 1, @p) 
extern Numb const*  code_get_numb(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_strg, 1p == 1, @p) 
extern char const*  code_get_strg(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_name, 1p == 1, @p) 
extern char const*  code_get_name(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_inst_count, pure)    
extern unsigned int code_get_inst_count(void) is_PURE;
extern void         code_clear_inst_count(void);
//lint -sem(        code_get_tuple, 1p == 1, @p) 
extern Tuple const* code_get_tuple(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_set, 1p == 1, @p) 
extern Set const*   code_get_set(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(         code_get_idxset, 1p == 1, @p) 
extern IdxSet const* code_get_idxset(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_entry, 1p == 1, @p)
extern Entry const* code_get_entry(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_term, 1p == 1, @p)
extern Term const*  code_get_term(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_size, 1p == 1, chneg(@)) 
extern int          code_get_size(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_bool, 1p == 1) 
extern bool         code_get_bool(CodeNode* node);
//lint -sem(        code_get_list, 1p == 1, @p) 
extern List const*  code_get_list(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_varclass, 1p == 1) 
extern VarClass     code_get_varclass(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_contype, 1p == 1) 
extern ConType      code_get_contype(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_rdef, 1p == 1, @p) 
extern RDef const*  code_get_rdef(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_rpar, 1p == 1, @p)
extern RPar const*  code_get_rpar(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_bits, 1p == 1) 
extern unsigned int code_get_bits(CodeNode* node) expects_NONNULL;
//lint -sem(        code_get_symbol, 1p == 1, @p) 
extern Symbol*      code_get_symbol(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_define, 1p == 1, @p) 
extern Define*      code_get_define(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_get_bound, 1p == 1, @p) 
extern Bound const* code_get_bound(CodeNode* node) expects_NONNULL returns_NONNULL;
//lint -sem(        code_value_numb, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_numb(CodeNode* node, Numb* numb) expects_NONNULL;
//lint -sem(        code_value_strg, inout(1), 1p == 1, custodial(2), 2p >= 1) 
extern void         code_value_strg(CodeNode* node, char const* strg) expects_NONNULL;
//lint -sem(        code_value_name, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_name(CodeNode* node, char const* name) expects_NONNULL;
//lint -sem(        code_value_tuple, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_tuple(CodeNode* node, Tuple* tuple) expects_NONNULL;
//lint -sem(        code_value_set, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_set(CodeNode* node, Set* set) expects_NONNULL;
//lint -sem(        code_value_idxset, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_idxset(CodeNode* node, IdxSet* idxset) expects_NONNULL;
//lint -sem(        code_value_entry, inout(1), 1p == 1, custodial(2), 2p == 1)  
extern void         code_value_entry(CodeNode* node, Entry* entry) expects_NONNULL;
//lint -sem(        code_value_term, inout(1), 1p == 1, custodial(2), 2p == 1)  
extern void         code_value_term(CodeNode* node, Term* term) expects_NONNULL;
//lint -sem(        code_value_steal_term, inout(1), 1p == 1, chneg(2)) 
extern Term*        code_value_steal_term(CodeNode* node, int no) expects_NONNULL;
//lint -sem(        code_value_bool, inout(1), 1p == 1) 
extern void         code_value_bool(CodeNode* node, bool bvalue) expects_NONNULL;
//lint -sem(        code_value_size, inout(1), 1p == 1, chneg(2)) 
extern void         code_value_size(CodeNode* node, int size) expects_NONNULL;
//lint -sem(        code_value_list, inout(1), 1p == 1, custodial(2), 2p == 1)  
extern void         code_value_list(CodeNode* node, List* list) expects_NONNULL;
//lint -sem(        code_value_varclass, inout(1), 1p == 1) 
extern void         code_value_varclass(CodeNode* node, VarClass varclass) expects_NONNULL;
//lint -sem(        code_value_contype, inout(1), 1p == 1) 
extern void         code_value_contype(CodeNode* node, ConType contype) expects_NONNULL;
//lint -sem(        code_value_rdef, inout(1), 1p == 1, custodial(2), 2p == 1)  
extern void         code_value_rdef(CodeNode* node, RDef* rdef) expects_NONNULL;
//lint -sem(        code_value_rpar, inout(1), 1p == 1, custodial(2), 2p == 1) 
extern void         code_value_rpar(CodeNode* node, RPar* rpar) expects_NONNULL;
//lint -sem(        code_value_bits, inout(1), 1p == 1) 
extern void         code_value_bits(CodeNode* node, unsigned int bits) expects_NONNULL;
//lint -sem(        code_value_bound, inout(1), 1p == 1, custodial(2), 2p == 1)  
extern void         code_value_bound(CodeNode* node, Bound* bound) expects_NONNULL;
//lint -sem(        code_value_void, inout(1), 1p == 1) 
extern void         code_value_void(CodeNode* node) expects_NONNULL;
//lint -sem(        code_copy_value, inout(1), 1p == 1, 2p == 1, @P >= malloc(1))  
extern void         code_copy_value(CodeNode* dst, CodeNode const* src) expects_NONNULL;

//lint -sem(        code_eval_child, 1p == 1, chneg(2), @p == 1) 
extern CodeNode*    code_eval_child(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_numb, 1p == 1, chneg(2), @p == 1) 
extern Numb const*  code_eval_child_numb(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_strg, 1p == 1, chneg(2), @p)  
extern char const*  code_eval_child_strg(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_name, 1p == 1, chneg(2), @p) 
extern char const*  code_eval_child_name(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_tuple, 1p == 1, chneg(2), @p == 1)  
extern Tuple const* code_eval_child_tuple(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_set, 1p == 1, chneg(2), @p == 1)  
extern Set const*   code_eval_child_set(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(         code_eval_child_idxset, 1p == 1, chneg(2), @p == 1)  
extern IdxSet const* code_eval_child_idxset(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_entry, 1p == 1, chneg(2), @p == 1)  
extern Entry const* code_eval_child_entry(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_term, 1p == 1, chneg(2), @p == 1)  
extern Term const*  code_eval_child_term(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_size, 1p == 1, chneg(2), chneg(@)) 
extern int          code_eval_child_size(CodeNode const* node, int no) expects_NONNULL;
//lint -sem(        code_eval_child_bool, 1p == 1, chneg(2)) 
extern bool         code_eval_child_bool(CodeNode const* node, int no) expects_NONNULL;
//lint -sem(        code_eval_child_list, 1p == 1, chneg(2), @p == 1)  
extern List const*  code_eval_child_list(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_varclass, 1p == 1, chneg(2)) 
extern VarClass     code_eval_child_varclass(CodeNode const* node, int no) expects_NONNULL;
//lint -sem(        code_eval_child_contype, 1p == 1, chneg(2)) 
extern ConType      code_eval_child_contype(CodeNode const* node, int no) expects_NONNULL;
//lint -sem(        code_eval_child_rdef, 1p == 1, chneg(2), @p == 1)  
extern RDef const*  code_eval_child_rdef(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_rpar, 1p == 1, chneg(2), @p == 1)  
extern RPar const*  code_eval_child_rpar(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_bits, 1p == 1, chneg(2)) 
extern unsigned int code_eval_child_bits(CodeNode const* node, int no) expects_NONNULL;
//lint -sem(        code_eval_child_symbol, 1p == 1, chneg(2), @p == 1)  
extern Symbol*      code_eval_child_symbol(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_define, 1p == 1, chneg(2), @p == 1)  
extern Define*      code_eval_child_define(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;
//lint -sem(        code_eval_child_bound, 1p == 1, chneg(2), @p == 1)  
extern Bound const* code_eval_child_bound(CodeNode const* node, int no) expects_NONNULL returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _CODE_H_ 
