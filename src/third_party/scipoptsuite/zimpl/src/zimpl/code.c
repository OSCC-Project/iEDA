/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: code.c                                                        */
/*   Name....: Code Node Functions                                           */
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <assert.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"
#include "zimpl/stkchk.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/idxset.h"
#include "zimpl/rdefpar.h"
#include "zimpl/bound.h"
#include "zimpl/define.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/list.h"
#include "zimpl/stmt.h"
#include "zimpl/local.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"

typedef union code_value CodeValue;

union code_value
{
   Numb*        numb;
   char const*  strg;
   char const*  name;
   Tuple*       tuple;
   Set*         set;
   Term*        term;
   Entry*       entry;
   IdxSet*      idxset;
   bool         bval;
   int          size;
   List*        list;
   VarClass     varclass;
   ConType      contype;
   RDef*        rdef;
   RPar*        rpar;
   unsigned int bits;
   Symbol*      sym;
   Define*      def;
   Bound*       bound;
};

#define MAX_CHILDS 8

struct code_node
{
   SID
   CodeType    type;
   Inst        eval;
   CodeValue   value;
   CodeNode*   child[MAX_CHILDS];
   Stmt const* stmt;
   int         column;
};

#define CODE_SID  0x436f6465

static CodeNode*    root       = NULL;
static unsigned int inst_count = 0;

bool code_is_valid(CodeNode const* node)
{
   return ((node != NULL) && SID_ok(node, CODE_SID));
}

CodeNode* code_new_inst(Inst inst, int childs, ...)
{
   va_list   ap;
   CodeNode* node = calloc(1, sizeof(*node));
   int       i;
   CodeNode* child;
   
   assert(inst   != INST_NULL);
   assert(node   != NULL);
   assert(childs <= MAX_CHILDS);

   node->type   = CODE_ERR; /* call to eval() will set this */
   node->eval   = inst;
   /* Das sollte eigendlich ein parameter sein, aber wozu bei jedem
    * code_new diese Funktionen mit uebergeben.
    */
   node->stmt   = scan_get_stmt();
   node->column = scan_get_column();
   
   SID_set(node, CODE_SID);
   assert(code_is_valid(node));

   /*lint -save -e826 */
   va_start(ap, childs);

   for(i = 0; i < childs; i++)
   {
      child = va_arg(ap, CodeNode*);
      assert((child == NULL) || code_is_valid(child));
      node->child[i] = child;
   }
   va_end(ap);
   /*lint -restore */
   
   return node;
}

/* We eat the numb, i.e. we will free it!
 */
CodeNode* code_new_numb(Numb* numb)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type       = CODE_NUMB;
   node->eval       = i_nop;
   node->value.numb = numb; 
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_strg(char const* strg)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(strg != NULL);
   assert(node != NULL);

   node->type       = CODE_STRG;
   node->eval       = i_nop;
   node->value.strg = strg;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_name(char const* name)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(name != NULL);
   assert(node != NULL);

   node->type       = CODE_NAME;
   node->eval       = i_nop;
   node->value.name = name;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_size(int size)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(size >= 0);
   assert(node != NULL);

   node->type       = CODE_SIZE;
   node->eval       = i_nop;
   node->value.size = size;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_varclass(VarClass varclass)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type           = CODE_VARCLASS;
   node->eval           = i_nop;
   node->value.varclass = varclass;
   node->stmt           = scan_get_stmt();
   node->column         = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_contype(ConType contype)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type          = CODE_CONTYPE;
   node->eval          = i_nop;
   node->value.contype = contype;
   node->stmt          = scan_get_stmt();
   node->column        = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_bits(unsigned int bits)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type       = CODE_BITS;
   node->eval       = i_nop;
   node->value.bits = bits;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_symbol(Symbol* sym)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type       = CODE_SYM;
   node->eval       = i_nop;
   node->value.sym  = sym;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

CodeNode* code_new_define(Define* def)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);

   node->type       = CODE_DEF;
   node->eval       = i_nop;
   node->value.def  = def;
   node->stmt       = scan_get_stmt();
   node->column     = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

/* We eat the bound, i.e. we will free it!
 */
CodeNode* code_new_bound(BoundType type)
{
   CodeNode* node = calloc(1, sizeof(*node));

   assert(node != NULL);
   assert(type == BOUND_INFTY || type == BOUND_MINUS_INFTY);

   node->type        = CODE_BOUND;
   node->eval        = i_nop;
   node->value.bound = bound_new(type, NULL); 
   node->stmt        = scan_get_stmt();
   node->column      = scan_get_column();

   SID_set(node, CODE_SID);
   assert(code_is_valid(node));
   
   return node;
}

void code_free_value(CodeNode* node)
{
   assert(code_is_valid(node));

   switch(node->type)
   {
   case CODE_ERR :
   case CODE_VOID :
      /* Kann passieren, wenn bei code_value_() ein bis dahin unbenutzter
       * Knoten verwendet wird.
       */
      break;
   case CODE_NUMB :
      assert(node->value.numb != NULL);
      numb_free(node->value.numb);
      break;
   case CODE_STRG :
   case CODE_NAME :
      break;
   case CODE_TUPLE :
      assert(node->value.tuple != NULL);
      tuple_free(node->value.tuple);
      node->value.tuple = NULL;
      node->type        = CODE_ERR;
      break;
   case CODE_SET :
      assert(node->value.set != NULL);
      set_free(node->value.set);      
      node->value.set = NULL;
      node->type      = CODE_ERR;
      break;
   case CODE_TERM :
      assert(node->value.term != NULL);
      term_free(node->value.term);
      node->value.term = NULL;
      node->type       = CODE_ERR;      
      break;
   case CODE_ENTRY :
      assert(node->value.entry != NULL);
      entry_free(node->value.entry);
      node->value.entry = NULL;
      node->type        = CODE_ERR;      
      break;
   case CODE_IDXSET :
      assert(node->value.idxset != NULL);
      idxset_free(node->value.idxset);
      node->value.entry = NULL;
      node->type        = CODE_ERR;      
      break;
   case CODE_BOOL :
   case CODE_SIZE :
      break;
   case CODE_LIST :
      assert(node->value.list != NULL);
      list_free(node->value.list);
      node->value.list = NULL;
      node->type       = CODE_ERR;      
      break;
   case CODE_VARCLASS :
   case CODE_CONTYPE :
      break;
   case CODE_RDEF :
      assert(node->value.rdef != NULL);
      rdef_free(node->value.rdef);
      node->value.rdef = NULL;
      node->type       = CODE_ERR;      
      break;
   case CODE_RPAR :
      assert(node->value.rpar != NULL);
      rpar_free(node->value.rpar);
      node->value.rpar = NULL;
      node->type       = CODE_ERR;      
      break;
   case CODE_BITS :
      break;
   case CODE_SYM :
      break;
   case CODE_DEF :
      break;
   case CODE_BOUND :
      assert(node->value.bound != NULL);
      bound_free(node->value.bound);
      break;
   default :
      abort();
   }
}

void code_free(CodeNode* node)
{
   int i;

   for(i = 0; i < MAX_CHILDS; i++)
      if (node->child[i] != NULL)
         code_free(node->child[i]);

   code_free_value(node);

   free(node);
}

void code_set_child(CodeNode* node, int idx, CodeNode* child)
{
   assert(code_is_valid(node));
   assert(idx   >= 0);
   assert(idx   <  MAX_CHILDS);
   assert(child != NULL);

   node->child[idx] = child;
}

CodeType code_get_type(CodeNode const* node)
{
   assert(code_is_valid(node));

   return node->type;
}

Inst code_get_inst(CodeNode const* node)
{
   assert(code_is_valid(node));

   return node->eval;
}
     
void code_set_root(CodeNode* node)
{
   assert(code_is_valid(node));
   
   root = node;
}

CodeNode* code_get_root(void)
{
   return root;
}

unsigned int code_get_inst_count()
{
   return inst_count;
}
     
void code_clear_inst_count()
{
   inst_count = 0;
}
     
static inline CodeNode* code_check_type(CodeNode* node, CodeType expected)
{
   static char const* const tname[] =
   {
      "Error", "Number", "String", "Name", "Tuple", "Set", "Term", "Bool", "Size",
      "IndexSet", "List", "Nothing", "Entry", "VarClass", "ConType",
      "ReadDefinition", "ReadParameter", "BitFlag", "Symbol", "Define", "Bound"
   };
   assert(code_is_valid(node));
   assert(sizeof(tname) / sizeof(tname[0]) > (size_t)node->type);
   
   if (node->type != expected)
   {
      assert(sizeof(tname) / sizeof(tname[0]) > (size_t)expected);
      
      fprintf(stderr, "*** Error 159: Type error, expected %s got %s\n",
         tname[expected], tname[node->type]);
      code_errmsg(node);
      zpl_exit(EXIT_FAILURE);
   }
   return node;
}

void code_errmsg(CodeNode const* node)
{
   fprintf(stderr, "*** File: %s Line %d\n",
      stmt_get_filename(node->stmt),
      stmt_get_lineno(node->stmt));

   show_source(stderr, stmt_get_text(node->stmt), node->column);

   if (verbose >= VERB_CHATTER)
      local_print_all(stderr);
}

CodeNode* code_eval(CodeNode* node)
{
   assert(code_is_valid(node));

   inst_count++;

   stkchk_used(); // record maximum stack use
   
   return (*node->eval)(node);
}


bool code_prune_tree(CodeNode* node)
{
   static Inst const prunable[] = 
   { 
      i_expr_abs, i_expr_sgn, i_expr_add, i_expr_card, i_expr_ceil, i_expr_div, i_expr_exp, 
      i_expr_sqrt, i_expr_fac, i_expr_floor, i_expr_if_else, i_expr_intdiv, i_expr_length, i_expr_ln,
      i_expr_log, i_expr_ord, i_expr_prod, i_expr_round, i_expr_sum, i_expr_max,
      i_expr_max2, i_expr_sglmax, i_expr_min, i_expr_min2, i_expr_sglmin, i_expr_mul, 
      i_expr_mod, i_expr_neg, i_expr_pow, i_expr_sub, i_expr_substr, NULL 
   };

   bool is_all_const = true;
   int  i;

   if (node->eval == (Inst)i_nop)
      return true;

   for(i = 0; i < MAX_CHILDS; i++)
   {
      if (node->child[i] != NULL)
      {
         bool is_const = code_prune_tree(node->child[i]);

         /* Writing directly && code_prune_tree might not work
          * due to shortcut evaluation.
          */
         is_all_const = is_all_const && is_const;
      }
   }

   if (is_all_const)
   {
      for(i = 0; prunable[i] != INST_NULL; i++)
         if (prunable[i] == node->eval)
            break;

      if (prunable[i] == INST_NULL)
         return false;

      (void)code_eval(node);

      for(i = 0; i < MAX_CHILDS; i++)
      {
         if (node->child[i] != NULL)
         {
            code_free(node->child[i]);
            node->child[i] = NULL;
         }
      }
      node->eval = i_nop;
   }   
   return is_all_const;
}


/* ----------------------------------------------------------------------------
 * Get Funktionen
 * ----------------------------------------------------------------------------
 */

CodeNode* code_get_child(CodeNode const* node, int no)
{
   assert(code_is_valid(node));
   assert(no              >= 0);
   assert(no              <  MAX_CHILDS);
   assert(node->child[no] != NULL);
   
   return node->child[no];
}

Numb const* code_get_numb(CodeNode* node)
{
   return code_check_type(node, CODE_NUMB)->value.numb;
}

char const* code_get_strg(CodeNode* node)
{
   return code_check_type(node, CODE_STRG)->value.strg;
}

char const* code_get_name(CodeNode* node)
{
   return code_check_type(node, CODE_NAME)->value.name;
}

Tuple const* code_get_tuple(CodeNode* node)
{
   return code_check_type(node, CODE_TUPLE)->value.tuple;
}

Set const* code_get_set(CodeNode* node)
{
   return code_check_type(node, CODE_SET)->value.set;
}

IdxSet const* code_get_idxset(CodeNode* node)
{
   return code_check_type(node, CODE_IDXSET)->value.idxset;
}

Entry const* code_get_entry(CodeNode* node)
{
   return code_check_type(node, CODE_ENTRY)->value.entry;
}

Term const* code_get_term(CodeNode* node)
{
   return code_check_type(node, CODE_TERM)->value.term;
}

int code_get_size(CodeNode* node)
{
   return code_check_type(node, CODE_SIZE)->value.size;
}

bool code_get_bool(CodeNode* node)
{
   return code_check_type(node, CODE_BOOL)->value.bval;
}

List const* code_get_list(CodeNode* node)
{
   return code_check_type(node, CODE_LIST)->value.list;
}

VarClass code_get_varclass(CodeNode* node)
{
   return code_check_type(node, CODE_VARCLASS)->value.varclass;
}

ConType code_get_contype(CodeNode* node)
{
   return code_check_type(node, CODE_CONTYPE)->value.contype;
}

RDef const* code_get_rdef(CodeNode* node)
{
   return code_check_type(node, CODE_RDEF)->value.rdef;
}

RPar const* code_get_rpar(CodeNode* node)
{
   return code_check_type(node, CODE_RPAR)->value.rpar;
}

unsigned int code_get_bits(CodeNode* node)
{
   return code_check_type(node, CODE_BITS)->value.bits;
}

Symbol* code_get_symbol(CodeNode* node)
{
   return code_check_type(node, CODE_SYM)->value.sym;
}

Define* code_get_define(CodeNode* node)
{
   return code_check_type(node, CODE_DEF)->value.def;
}

Bound const* code_get_bound(CodeNode* node)
{
   return code_check_type(node, CODE_BOUND)->value.bound;
}

/* ----------------------------------------------------------------------------
 * Value Funktionen
 * ----------------------------------------------------------------------------
 */
void code_value_numb(CodeNode* node, Numb* numb)
{
   assert(code_is_valid(node));

   code_free_value(node);
   
   node->type       = CODE_NUMB;
   node->value.numb = numb;
}

void code_value_strg(CodeNode* node, char const* strg)
{
   assert(code_is_valid(node));
   assert(strg != NULL);
   
   code_free_value(node);

   node->type       = CODE_STRG;
   node->value.strg = strg;
}

void code_value_name(CodeNode* node, char const* name)
{
   assert(code_is_valid(node));
   assert(name != NULL);
   
   code_free_value(node);

   node->type       = CODE_NAME;
   node->value.name = name;
}

void code_value_tuple(CodeNode* node, Tuple* tuple)
{
   assert(code_is_valid(node));
   assert(tuple_is_valid(tuple));

   code_free_value(node);

   node->type        = CODE_TUPLE;
   node->value.tuple = tuple;
}

void code_value_set(CodeNode* node, Set* set)
{
   assert(code_is_valid(node));
   assert(set_is_valid(set));

   code_free_value(node);

   node->type      = CODE_SET;
   node->value.set = set;
}

void code_value_idxset(CodeNode* node, IdxSet* idxset)
{
   assert(code_is_valid(node));
   assert(idxset_is_valid(idxset));

   code_free_value(node);

   node->type         = CODE_IDXSET;
   node->value.idxset = idxset;
}

void code_value_entry(CodeNode* node, Entry* entry)
{
   assert(code_is_valid(node));
   assert(entry_is_valid(entry));

   code_free_value(node);

   node->type        = CODE_ENTRY;
   node->value.entry = entry;
}

void code_value_term(CodeNode* node, Term* term)
{
   assert(code_is_valid(node));
   assert(term_is_valid(term));

   code_free_value(node);

   node->type       = CODE_TERM;
   node->value.term = term;
}

Term* code_value_steal_term(CodeNode* node, int no)
{
   CodeNode* child = code_get_child(node, no);

   assert(code_is_valid(node));
   assert(code_get_type(child) == CODE_TERM);

   code_free_value(node);

   node->type        = CODE_TERM;
   node->value.term  = child->value.term;

   child->type       = CODE_ERR;
   child->value.term = NULL;

   return node->value.term;
}


void code_value_bool(CodeNode* node, bool bval)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type       = CODE_BOOL;
   node->value.bval = bval;
}

void code_value_size(CodeNode* node, int size)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type       = CODE_SIZE;
   node->value.size = size;
}

void code_value_list(CodeNode* node, List* list)
{
   assert(code_is_valid(node));
   assert(list_is_valid(list));

   code_free_value(node);

   node->type       = CODE_LIST;
   node->value.list = list;
}

void code_value_varclass(CodeNode* node, VarClass varclass)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type           = CODE_VARCLASS;
   node->value.varclass = varclass;
}

void code_value_contype(CodeNode* node, ConType contype)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type          = CODE_CONTYPE;
   node->value.contype = contype;
}

void code_value_rdef(CodeNode* node, RDef* rdef)
{
   assert(code_is_valid(node));
   assert(rdef_is_valid(rdef));

   code_free_value(node);

   node->type       = CODE_RDEF;
   node->value.rdef = rdef;
}

void code_value_rpar(CodeNode* node, RPar* rpar)
{
   assert(code_is_valid(node));
   assert(rpar_is_valid(rpar));

   code_free_value(node);

   node->type       = CODE_RPAR;
   node->value.rpar = rpar;
}

void code_value_bits(CodeNode* node, unsigned int bits)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type       = CODE_BITS;
   node->value.bits = bits;
}

void code_value_bound(CodeNode* node, Bound* bound)
{
   assert(code_is_valid(node));

   code_free_value(node);
   
   node->type        = CODE_BOUND;
   node->value.bound = bound;
}

void code_value_void(CodeNode* node)
{
   assert(code_is_valid(node));

   code_free_value(node);

   node->type = CODE_VOID;
}

void code_copy_value(CodeNode* dst, CodeNode const* src)
{
   assert(code_is_valid(dst));
   assert(code_is_valid(src));
   
   code_free_value(dst);

   switch(src->type)
   {
   case CODE_NUMB :
      dst->value.numb = numb_copy(src->value.numb);
      break;
   case CODE_STRG :
      dst->value.strg = src->value.strg;
      break;
   case CODE_NAME :
      dst->value.name = src->value.name;
      break;
   case CODE_TUPLE:
      dst->value.tuple = tuple_copy(src->value.tuple);
      break;
   case CODE_SET :
      dst->value.set = set_copy(src->value.set);
      break;
   case CODE_IDXSET :
      dst->value.idxset = idxset_copy(src->value.idxset);
      break;
   case CODE_ENTRY :
      dst->value.entry = entry_copy(src->value.entry);
      break;
   case CODE_TERM :
      dst->value.term = term_copy(src->value.term);
      break;
   case CODE_BOOL :
      dst->value.bval = src->value.bval;
      break;
   case CODE_SIZE :
      dst->value.size = src->value.size;
      break;
   case CODE_LIST :
      dst->value.list = list_copy(src->value.list);
      break;
   case CODE_VARCLASS :
      dst->value.varclass = src->value.varclass;
      break;
   case CODE_CONTYPE :
      dst->value.contype = src->value.contype;
      break;
   case CODE_RDEF :
      dst->value.rdef = rdef_copy(src->value.rdef);
      break;
   case CODE_RPAR :
      dst->value.rpar = rpar_copy(src->value.rpar);
      break;
   case CODE_BITS :
      dst->value.bits = src->value.bits;
      break;
   case CODE_BOUND :
      dst->value.bound = bound_copy(src->value.bound);
      break;
   case CODE_VOID :
      break;
   default :
      abort();
   }
   dst->type = src->type;
}

/* ----------------------------------------------------------------------------
 * Macro Funktionen for better inlining
 * ----------------------------------------------------------------------------
 */
CodeNode* code_eval_child(CodeNode const* node, int no)
{
   return code_eval(code_get_child(node, no));
}

Numb const* code_eval_child_numb(CodeNode const* node, int no)
{
   return code_get_numb(code_eval(code_get_child(node, no)));
}

char const* code_eval_child_strg(CodeNode const* node, int no)
{
   return code_get_strg(code_eval(code_get_child(node, no)));
}

char const*code_eval_child_name(CodeNode const* node, int no)
{
   return code_get_name(code_eval(code_get_child(node, no)));
}

Tuple const* code_eval_child_tuple(CodeNode const* node, int no)
{
   return code_get_tuple(code_eval(code_get_child(node, no)));
}

Set const* code_eval_child_set(CodeNode const* node, int no)
{
   return code_get_set(code_eval(code_get_child(node, no)));
}

IdxSet const* code_eval_child_idxset(CodeNode const* node, int no)
{
   return code_get_idxset(code_eval(code_get_child(node, no)));
}

Entry const* code_eval_child_entry(CodeNode const* node, int no)
{
   return code_get_entry(code_eval(code_get_child(node, no)));
}

Term const* code_eval_child_term(CodeNode const* node, int no)
{
   return code_get_term(code_eval(code_get_child(node, no)));
}

int code_eval_child_size(CodeNode const* node, int no)
{
   return code_get_size(code_eval(code_get_child(node, no)));
}

bool code_eval_child_bool(CodeNode const* node, int no)
{
   return code_get_bool(code_eval(code_get_child(node, no)));
}

List const* code_eval_child_list(CodeNode const* node, int no)
{
   return code_get_list(code_eval(code_get_child(node, no)));
}

VarClass code_eval_child_varclass(CodeNode const* node, int no)
{
   return code_get_varclass(code_eval(code_get_child(node, no)));
}

ConType code_eval_child_contype(CodeNode const* node, int no)
{
   return code_get_contype(code_eval(code_get_child(node, no)));
}

RDef const* code_eval_child_rdef(CodeNode const* node, int no)
{
   return code_get_rdef(code_eval(code_get_child(node, no)));
}

RPar const* code_eval_child_rpar(CodeNode const* node, int no)
{
   return code_get_rpar(code_eval(code_get_child(node, no)));
}

unsigned int code_eval_child_bits(CodeNode const* node, int no)
{
   return code_get_bits(code_eval(code_get_child(node, no)));
}

Symbol* code_eval_child_symbol(CodeNode const* node, int no)
{
   return code_get_symbol(code_eval(code_get_child(node, no)));
}

Define* code_eval_child_define(CodeNode const* node, int no)
{
   return code_get_define(code_eval(code_get_child(node, no)));
}

Bound const* code_eval_child_bound(CodeNode const* node, int no)
{
   return code_get_bound(code_eval(code_get_child(node, no)));
}








