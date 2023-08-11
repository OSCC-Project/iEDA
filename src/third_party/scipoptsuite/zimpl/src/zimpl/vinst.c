/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: vinst.c                                                       */
/*   Name....: Variable Instructions                                         */
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
#include <stdbool.h>
#include <assert.h>

/* #define TRACE  1  */

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/define.h"
#include "zimpl/bound.h"
#include "zimpl/idxset.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/rdefpar.h"
#include "zimpl/conname.h"
#include "zimpl/stmt.h"
#include "zimpl/prog.h"
#include "zimpl/local.h"
#include "zimpl/list.h"
#include "zimpl/entry.h"
#include "zimpl/heap.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"
#include "zimpl/xlpglue.h"
#include "zimpl/strstore.h"

enum vbool_cmp_operator { VBOOL_LT, VBOOL_LE, VBOOL_EQ, VBOOL_NE, VBOOL_GE, VBOOL_GT };
enum vbool_fixed_result { VBOOL_TRUE, VBOOL_FALSE, VBOOL_OPEN };

typedef enum vbool_cmp_operator VBCmpOp;
typedef enum vbool_fixed_result VBFixed;

static int internal_vars = 0;
static int internal_cons = 0;

void interns_init()
{
   assert(internal_vars == 0);
   assert(internal_cons == 0);
}

void interns_exit()
{
   internal_vars = 0;
   internal_cons = 0;
}

//lint -sem(create_new_constraint, 1p, 2p, custodial(3), inout(3), 3p == 1, 5p == 1)
static void create_new_constraint(
   char const*  basename,
   char const*  extension,
   Term*        term,
   ConType      con_type,
   Numb const*  lrhs,
   unsigned int flags)
{
   char* cname;
   
   Trace("create_new_constraint");

   assert(basename  != NULL);
   assert(extension != NULL);
   assert(con_type  != CON_RANGE);
   assert(term_is_valid(term));
   assert(numb_is_valid(lrhs));
   
   cname = malloc(strlen(basename) + strlen(extension) + 10 + 1);
   sprintf(cname, "%s%s_%d", basename, extension, internal_cons++);
   (void)xlp_addcon_term(prog_get_lp(), cname, con_type, lrhs, lrhs, flags, term);

   term_free(term);
   free(cname);
}

static Entry* create_new_var_entry(
   char const*  basename,
   char const*  extension,
   VarClass     var_class,
   Bound const* lower,
   Bound const* upper)
{
   char*  vname;
   Tuple* tuple;
   Entry* entry;
   Var*   var;
   
   Trace("create_new_var_entry");

   assert(basename  != NULL);
   assert(extension != NULL);
   /*   assert(var_class != VAR_CON);*/
   assert(bound_is_valid(lower));
   assert(bound_is_valid(upper));
   
   vname = malloc(strlen(basename) + strlen(extension) + strlen(SYMBOL_NAME_INTERNAL) + 16);
   sprintf(vname, "%s%s%s_%d", SYMBOL_NAME_INTERNAL, basename, extension, internal_vars++);
   var   = xlp_addvar(prog_get_lp(), vname, var_class, lower, upper, numb_zero(), numb_zero());
   tuple = tuple_new(1);
   tuple_set_elem(tuple, 0, elem_new_strg(str_new(vname + strlen(SYMBOL_NAME_INTERNAL))));
   entry = entry_new_var(tuple, var);

   tuple_free(tuple);
   free(vname);

   return entry;
}

static VBFixed check_how_fixed(
   VBCmpOp     cmp_op,
   Numb const* rhs)
{
   VBFixed result = VBOOL_FALSE;
   
   switch(cmp_op)
   {
   case VBOOL_EQ :
      if (numb_equal(rhs, numb_zero()))
         result = VBOOL_TRUE;
      break;
   case VBOOL_NE :
      if (!numb_equal(rhs, numb_zero()))
         result = VBOOL_TRUE;
      break;
   case VBOOL_LE :
      if (numb_cmp(numb_zero(), rhs) <= 0)
         result = VBOOL_TRUE;
      break;
   case VBOOL_GE :
      if (numb_cmp(numb_zero(), rhs) >= 0)
         result = VBOOL_TRUE;
      break;
   case VBOOL_LT :
      if (numb_cmp(numb_zero(), rhs) < 0)
         result = VBOOL_TRUE;
      break;
   case VBOOL_GT :
      if (numb_cmp(numb_zero(), rhs) > 0)
         result = VBOOL_TRUE;
      break;
   default :
      abort();
   }
   return result;
}

static VBFixed check_if_fixed(
   VBCmpOp     cmp_op,
   Numb const* lower,
   Numb const* upper)
{
   VBFixed result = VBOOL_OPEN;

   /* lower <= upper
    */
   assert(numb_cmp(lower, upper) <= 0);

   switch(cmp_op)
   {
   case VBOOL_EQ :
      if (  (numb_cmp(lower, numb_zero()) > 0)
         || (numb_cmp(upper, numb_zero()) < 0))
         result = VBOOL_FALSE;
      else
      {
         if (numb_equal(lower, upper))
         {
            assert(numb_equal(lower, numb_zero()));
            assert(numb_equal(upper, numb_zero()));

            result = VBOOL_TRUE;
         }
      }
      break;
   case VBOOL_NE :
      if (  (numb_cmp(lower, numb_zero()) > 0)
         || (numb_cmp(upper, numb_zero()) < 0))
         result = VBOOL_TRUE;
      else
      {
         if (numb_equal(lower, upper))
         {
            assert(numb_equal(lower, numb_zero()));
            assert(numb_equal(upper, numb_zero()));
            
            result = VBOOL_FALSE;
         }
      }
      break;
   case VBOOL_LE :
      if (numb_cmp(upper, numb_zero()) <= 0)
         result = VBOOL_TRUE;
      else if (numb_cmp(lower, numb_zero()) > 0)
         result = VBOOL_FALSE;
      break;
   case VBOOL_GE :
      if (numb_cmp(lower, numb_zero()) >= 0)
         result = VBOOL_TRUE;
      else if (numb_cmp(upper, numb_zero()) < 0)
         result = VBOOL_FALSE;
      break;
   case VBOOL_LT :
      if (numb_cmp(upper, numb_zero()) < 0)
         result = VBOOL_TRUE;
      else if (numb_cmp(lower, numb_zero()) >= 0)
         result = VBOOL_FALSE;
      break;
   case VBOOL_GT :
      if (numb_cmp(lower, numb_zero()) > 0)
         result = VBOOL_TRUE;
      else if (numb_cmp(upper, numb_zero()) <= 0)
         result = VBOOL_FALSE;
      break;
   default :
      abort();
   }
   return result;
}


static CodeNode* handle_vbool_cmp(CodeNode* self, VBCmpOp cmp_op)
{
   Symbol*      sym;
   Term*        term;
   Term const*  term_lhs;
   Term const*  term_rhs;
   Numb*        rhs;
   unsigned int flags;
   char const*  cname;
   Bound*       lower;
   Bound*       upper;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_result;
   VBFixed      fixed = VBOOL_OPEN;
   
   Trace("handle_vbool_cmp");
   
   assert(code_is_valid(self));

   term_lhs   = code_eval_child_term(self, 0);
   term_rhs   = code_eval_child_term(self, 1);
   flags      = 0;

   rhs        = numb_new_sub(term_get_constant(term_rhs), term_get_constant(term_lhs));
   term       = term_sub_term(term_lhs, term_rhs);

   if (!term_is_all_integer(term))
   {
      fprintf(stderr, "*** Error 177: Boolean constraint not all integer\n");
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   }
   /* Symbol for internal entries
    */
   sym        = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);

   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   lower      = term_get_lower_bound(term);
   upper      = term_get_upper_bound(term);
   
   if (bound_get_type(lower) != BOUND_VALUE || bound_get_type(upper) != BOUND_VALUE)
   {
      fprintf(stderr, "*** Error 185: Term in Boolean constraint not bounded\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   
   /* Check if trival infeasible
    */
   if (term_get_elements(term) == 0)
   {
      if (stmt_trigger_warning(176))
      {
         fprintf(stderr, "--- Warning 176: Empty LHS, in Boolean constraint\n");
         code_errmsg(code_get_child(self, 0));
      }
      fixed = check_how_fixed(cmp_op, rhs);

      assert(fixed != VBOOL_OPEN);
   }
   else
   {
      fixed = check_if_fixed(cmp_op, bound_get_value(lower), bound_get_value(upper));
   }
   if (fixed != VBOOL_OPEN)
   {
      if (fixed == VBOOL_TRUE)
         entry_result = create_new_var_entry(cname, "_re", VAR_INT, bound_one, bound_one);
      else
      {
         assert(fixed == VBOOL_FALSE);

         entry_result = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_zero);
      }
      if (stmt_trigger_warning(178))
      {
         fprintf(stderr,
            "--- Warning 178: Conditional always true or false due to bounds\n");
         code_errmsg(code_get_child(self, 0)); /* pos of warning message */
      }
      term_free(term);
   }
   else
   {
      Entry*       entry_xplus;
      Entry*       entry_xminus;
      Entry*       entry_bplus;
      Entry*       entry_bminus;
      
      /* Remove constant from term
       */
      term_add_constant(term, rhs);

      if (numb_cmp(bound_get_value(lower), numb_zero()) < 0)
      {
         Numb* numb = numb_copy(bound_get_value(lower));
         numb_abs(numb);
         bound_free(lower);
         lower = bound_new(BOUND_VALUE, numb);
         numb_free(numb);
      }
      else
      {
         bound_free(lower);
         lower = bound_new(BOUND_VALUE, numb_zero());
      }
   
      if (numb_cmp(bound_get_value(upper), numb_zero()) < 0)
      {
         bound_free(upper);
         upper = bound_new(BOUND_VALUE, numb_zero());
      }
      /* Create x^+, x^-, b^+, b^-, Result
       */
      entry_xplus  = create_new_var_entry(cname, "_xp", VAR_INT, bound_zero, upper);
      entry_xminus = create_new_var_entry(cname, "_xm", VAR_INT, bound_zero, lower);
      entry_bplus  = create_new_var_entry(cname, "_bp", VAR_INT, bound_zero, bound_one);
      entry_bminus = create_new_var_entry(cname, "_bm", VAR_INT, bound_zero, bound_one);
      entry_result = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_one);

      /* add term = x^+ + x^-
       */
      term_add_elem(term, entry_xplus, numb_minusone(), MFUN_NONE);
      term_add_elem(term, entry_xminus, numb_one(), MFUN_NONE);

      create_new_constraint(cname, "_a", term, CON_EQUAL, rhs, flags);
   
      /* bplus <= xplus */
      term = term_new(2);
      term_add_elem(term, entry_bplus, numb_one(), MFUN_NONE);
      term_add_elem(term, entry_xplus, numb_minusone(), MFUN_NONE);
      create_new_constraint(cname, "_b", term, CON_RHS, numb_zero(), flags);

      /* bminus <= xminus */
      term = term_new(2);
      term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
      term_add_elem(term, entry_xminus, numb_minusone(), MFUN_NONE);
      create_new_constraint(cname, "_c", term, CON_RHS, numb_zero(), flags);

      /* bplus * upper >= xplus */
      term = term_new(2);
      if (!numb_equal(bound_get_value(upper), numb_zero()))
         term_add_elem(term, entry_bplus, bound_get_value(upper), MFUN_NONE);
      term_add_elem(term, entry_xplus, numb_minusone(), MFUN_NONE);
      create_new_constraint(cname, "_d", term, CON_LHS, numb_zero(), flags);

      /* bminus * lower >= xminus */
      term = term_new(2);
      if (!numb_equal(bound_get_value(lower), numb_zero()))
         term_add_elem(term, entry_bminus, bound_get_value(lower), MFUN_NONE);
      term_add_elem(term, entry_xminus, numb_minusone(), MFUN_NONE);
      create_new_constraint(cname, "_e", term, CON_LHS, numb_zero(), flags);

      switch(cmp_op)
      {
      case VBOOL_EQ :
         /* 1 - result = bplus + bminus =>  result + bplus + bminus = 1 */
         term = term_new(3);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_EQUAL, numb_one(), flags);
         break;
      case VBOOL_NE :
         /* result = bplus + bminus  =>  result - bplus - bminus = 0*/
         term = term_new(3);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bplus,  numb_minusone(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_minusone(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_EQUAL, numb_zero(), flags);
         break;
      case VBOOL_LE :
         /* bplus + bminus <= 1,
          * result = 1 - bplus => result + bplus = 1
          */
         term = term_new(2);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_RHS, numb_one(), flags);

         term = term_new(2);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_g", term, CON_EQUAL, numb_one(), flags);
         break;
      case VBOOL_GE :
         /* bplus + bminus <= 1,
          * result = 1 - bminus => result + bminus = 1
          */
         term = term_new(2);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_RHS, numb_one(), flags);

         term = term_new(2);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus,  numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_g", term, CON_EQUAL, numb_one(), flags);
         break;
      case VBOOL_LT :
         /* bplus + bminus <= 1,
          * result = bminus => result - bminus = 0
          */
         term = term_new(2);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_RHS, numb_one(), flags);

         /* This is somewhat superflous
          */
         term = term_new(2);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus,  numb_minusone(), MFUN_NONE);
         create_new_constraint(cname, "_g", term, CON_EQUAL, numb_zero(), flags);
         break;
      case VBOOL_GT :
         /* bplus + bminus <= 1,
          * result = bplus => result - bplus = 0
          */
         term = term_new(2);
         term_add_elem(term, entry_bplus,  numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bminus, numb_one(), MFUN_NONE);
         create_new_constraint(cname, "_f", term, CON_RHS, numb_one(), flags);

         /* This is somewhat superflous
          */
         term = term_new(2);
         term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
         term_add_elem(term, entry_bplus,  numb_minusone(), MFUN_NONE);
         create_new_constraint(cname, "_g", term, CON_EQUAL, numb_zero(), flags);
         break;
      default :
         abort();
      }
      symbol_add_entry(sym, entry_xplus);
      symbol_add_entry(sym, entry_xminus);
      symbol_add_entry(sym, entry_bplus);
      symbol_add_entry(sym, entry_bminus);
   }
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);
   
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_one);
   bound_free(bound_zero);
   bound_free(lower);
   bound_free(upper);
   numb_free(rhs);
   
   return self;
}

CodeNode* i_vbool_ne(CodeNode* self)
{
   Trace("i_vbool_ne");

   return handle_vbool_cmp(self, VBOOL_NE);
}

CodeNode* i_vbool_eq(CodeNode* self)
{
   Trace("i_vbool_eq");

   return handle_vbool_cmp(self, VBOOL_EQ);
}

CodeNode* i_vbool_lt(CodeNode* self)
{
   Trace("i_vbool_lt");

   return handle_vbool_cmp(self, VBOOL_LT);
}

CodeNode* i_vbool_le(CodeNode* self)
{
   Trace("i_vbool_le");

   return handle_vbool_cmp(self, VBOOL_LE);
}

CodeNode* i_vbool_gt(CodeNode* self)
{
   Trace("i_vbool_gt");

   return handle_vbool_cmp(self, VBOOL_GT);
}

CodeNode* i_vbool_ge(CodeNode* self)
{
   Trace("i_vbool_ge");

   return handle_vbool_cmp(self, VBOOL_GE);
}

CodeNode* i_vbool_and(CodeNode* self)
{   
   Term const*  term_a;
   Term const*  term_b;
   char const*  cname;
   Term*        term;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_result;
   unsigned int flags = 0;
   Symbol*      sym;
   
   Trace("i_vbool_and");

   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   term_a     = code_eval_child_term(self, 0);
   term_b     = code_eval_child_term(self, 1);

   assert(term_get_elements(term_a) == 1);
   assert(term_get_elements(term_b) == 1);

   entry_result  = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_one);

   /* ------------------------------------- */

   /* a - re >= 0 */
   term = term_copy(term_a);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_a", term, CON_LHS, numb_zero(), flags);

   /* b - re >= 0 */
   term = term_copy(term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_b", term, CON_LHS, numb_zero(), flags);
   
   /* a + b - re <= 1 */
   term = term_add_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_c", term, CON_RHS, numb_one(), flags);
   
   /* ------------------------------------- */
   
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   /* Symbol for internal entries
    */
   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_zero);
   bound_free(bound_one);
   
   return self;
}

CodeNode* i_vbool_or(CodeNode* self)
{
   Term const*  term_a;
   Term const*  term_b;
   char const*  cname;
   Term*        term;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_result;
   unsigned int flags = 0;
   Symbol*      sym;
   
   Trace("i_vbool_or");

   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   term_a     = code_eval_child_term(self, 0);
   term_b     = code_eval_child_term(self, 1);

   assert(term_get_elements(term_a) == 1);
   assert(term_get_elements(term_b) == 1);

   entry_result  = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_one);

   /* ------------------------------------- */

   /* a - re <= 0 */
   term = term_copy(term_a);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_a", term, CON_RHS, numb_zero(), flags);

   /* b - re <= 0 */
   term = term_copy(term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_b", term, CON_RHS, numb_zero(), flags);
   
   /* a + b - re >= 0 */
   term = term_add_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_c", term, CON_LHS, numb_zero(), flags);
   
   /* ------------------------------------- */
   
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   /* Symbol for internal entries
    */
   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_zero);
   bound_free(bound_one);
   
   return self;
}

CodeNode* i_vbool_xor(CodeNode* self)
{
   Term const*  term_a;
   Term const*  term_b;
   char const*  cname;
   Term*        term;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_result;
   unsigned int flags = 0;
   Symbol*      sym;
   Numb*        numb;
   
   Trace("i_vbool_xor");

   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   term_a     = code_eval_child_term(self, 0);
   term_b     = code_eval_child_term(self, 1);

   assert(term_get_elements(term_a) == 1);
   assert(term_get_elements(term_b) == 1);

   entry_result  = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_one);

   /* ------------------------------------- */

   /* a + b - re >= 0 */
   term = term_add_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_a", term, CON_LHS, numb_zero(), flags);

   /* a - b - re <= 0 */
   term = term_sub_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_b", term, CON_RHS, numb_zero(), flags);
   
   /* a - b + re >= 0 */
   term = term_sub_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
   create_new_constraint(cname, "_c", term, CON_LHS, numb_zero(), flags);
   
   /* a + b + re <= 2 */
   numb = numb_new_integer(2);
   term = term_add_term(term_a, term_b);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
   create_new_constraint(cname, "_d", term, CON_RHS, numb, flags);
   numb_free(numb);
   
   /* ------------------------------------- */
   
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   /* Symbol for internal entries
    */
   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_zero);
   bound_free(bound_one);
   
   return self;
}

CodeNode* i_vbool_not(CodeNode* self)
{
   Term const*  term_a;
   char const*  cname;
   Term*        term;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_result;
   unsigned int flags = 0;
   Symbol*      sym;
   
   Trace("i_vbool_not");

   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   term_a     = code_eval_child_term(self, 0);

   assert(term_get_elements(term_a) == 1);

   entry_result  = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bound_one);

   /* ------------------------------------- */

   /* a + re == 1 */
   term = term_copy(term_a);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
   create_new_constraint(cname, "_a", term, CON_EQUAL, numb_one(), flags);
   
   /* ------------------------------------- */
   
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   /* Symbol for internal entries
    */
   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_zero);
   bound_free(bound_one);
   
   return self;
}


static void generate_conditional_constraint(
   CodeNode const* self,
   Term const*     vif_term,
   Term const*     lhs_term,
   ConType         con_type,
   Numb const*     rhs,
   unsigned int    flags,
   bool            then_case)
{
   Bound*       bound;
   Numb const*  bound_val;

   Trace("generate_conditional_constraint");

   assert(con_type == CON_RHS || con_type == CON_LHS);

   bound = (con_type == CON_RHS)
      ? term_get_upper_bound(lhs_term)
      : term_get_lower_bound(lhs_term);
   
   if (bound_get_type(bound) != BOUND_VALUE)
   {
      fprintf(stderr, "*** Error 179: Conditional only possible on bounded constraints\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   bound_val = bound_get_value(bound);

   if (  (con_type == CON_RHS && numb_cmp(bound_val, rhs) <= 0)
      || (con_type == CON_LHS && numb_cmp(bound_val, rhs) >= 0))
   {
      /* ??? This can be triggered for an eqiality constraint if after the
       * split one part is always true.
       */
      if (stmt_trigger_warning(180))
      {
         fprintf(stderr, "--- Warning 180: Conditional constraint always true due to bounds\n");

         code_errmsg(self);
      }  
   }
   else
   {
      char const*  basename  = conname_get();
      char*        cname     = malloc(strlen(basename) + 5);
      Term*        big_term  = term_copy(vif_term);
      Numb*        big_m     = then_case ? numb_new_sub(bound_val, rhs) : numb_new_sub(rhs, bound_val);
      Numb const*  new_rhs   = then_case ? bound_val : rhs;
      
      term_mul_coeff(big_term, big_m);

      term_append_term(big_term, lhs_term);

      sprintf(cname, "%s_%c_%c", basename,
         then_case           ? 't' : 'e',
         con_type == CON_RHS ? 'r' : 'l');

      (void)xlp_addcon_term(prog_get_lp(), cname, con_type, new_rhs, new_rhs, flags, big_term);

      numb_free(big_m);
      term_free(big_term);
      free(cname);
   }
   bound_free(bound);
}

/*ARGSUSED*/
static void generate_indicator_constraint(
   is_UNUSED CodeNode const* self,
   Term const*            vif_term,
   Term const*            lhs_term,
   ConType                con_type,
   Numb const*            rhs,
   unsigned int           flags,
   bool                   then_case)
{
   Numb*        lhs;
   Term*        ind_term;
   char const*  basename;
   char*        cname;

   Trace("generate_indicator_constraint");

   assert(flags & LP_FLAG_CON_INDIC);

   ind_term = term_make_conditional(vif_term, lhs_term, then_case);

   basename = conname_get();
   cname    = malloc(strlen(basename) + 5);
   sprintf(cname, "%s_%c", basename, then_case ? 't' : 'f');

   /* CON_EQUAL -> lhs == rhs
    * CON_RHS   -> lhs == rhs, lhs does not matter
    * CON_LHS   -> lhs == rhs, rhs does not matter
    * CON_RANGE -> not allowed
    */
   lhs = numb_copy(rhs);

#if 0   
   if (con_type == CON_LHS)
      numb_neg(lhs);
#endif

   (void)xlp_addcon_term(prog_get_lp(), cname, con_type, lhs, rhs, flags, ind_term);
   
   term_free(ind_term);
   numb_free(lhs);
   free(cname);
}


static void handle_vif_then_else(
   CodeNode const* self,
   Term const*     vif_term,
   Term const*     lhs_term,
   ConType         con_type,
   Term const*     rhs_term,
   unsigned int    flags,
   bool            then_case)
{
   Term*        term;
   Numb*        rhs;
   
   Trace("handle_vif_then_else");
   
   rhs   = numb_new_sub(term_get_constant(rhs_term), term_get_constant(lhs_term));
   term  = term_sub_term(lhs_term, rhs_term);
   term_add_constant(term, rhs);

   /* Check if trival infeasible
    */
   if (term_get_elements(term) == 0)
   {
      fprintf(stderr, "*** Error 181: Empty LHS, not allowed in conditional constraint\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(con_type == CON_RHS || con_type == CON_LHS || con_type == CON_EQUAL);

   if (!term_is_linear(term))
   {
      fprintf(stderr, "*** Error 222: Term inside a then or else constraint not linear\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
        
   if (flags & LP_FLAG_CON_INDIC)
      generate_indicator_constraint(self, vif_term, term, con_type, rhs, flags, then_case);
   else
   {
      /* <=, ==
       */
      if (con_type == CON_RHS || con_type == CON_EQUAL)
         generate_conditional_constraint(self, vif_term, term, CON_RHS, rhs, flags, then_case);
   
      /* >=, ==
       */
      if (con_type == CON_LHS || con_type == CON_EQUAL)
         generate_conditional_constraint(self, vif_term, term, CON_LHS, rhs, flags, then_case);
   }

   numb_free(rhs);
   term_free(term);
}

CodeNode* i_vif_else(CodeNode* self)
{
   Term const*  vif_term;
   Term const*  lhs_term;
   Term const*  rhs_term;
   ConType      con_type;   
   unsigned int flags;
   
   Trace("i_vif_else");
   
   assert(code_is_valid(self));

   vif_term   = code_eval_child_term(self, 0);
   flags      = code_eval_child_bits(self, 7);

   lhs_term   = code_eval_child_term(self, 1);
   con_type   = code_eval_child_contype(self, 2);
   rhs_term   = code_eval_child_term(self, 3);

   handle_vif_then_else(code_get_child(self, 1), /* pos for error messages */
      vif_term, lhs_term, con_type, rhs_term, flags, true);
   
   lhs_term   = code_eval_child_term(self, 4);
   con_type   = code_eval_child_contype(self, 5);
   rhs_term   = code_eval_child_term(self, 6);

   handle_vif_then_else(code_get_child(self, 4), /* pos for error messages */
      vif_term, lhs_term, con_type, rhs_term, flags, false);

   code_value_void(self);

   conname_next();

   return self;
}

CodeNode* i_vif(CodeNode* self)
{
   Term const*  vif_term;
   Term const*  lhs_term;
   Term const*  rhs_term;
   ConType      con_type;   
   unsigned int flags;
   
   Trace("i_vif");
   
   assert(code_is_valid(self));

   vif_term   = code_eval_child_term(self, 0);
   lhs_term   = code_eval_child_term(self, 1);
   con_type   = code_eval_child_contype(self, 2);
   rhs_term   = code_eval_child_term(self, 3);
   flags      = code_eval_child_bits(self, 4);

   handle_vif_then_else(code_get_child(self, 1), /* pos for error messages */
      vif_term, lhs_term, con_type, rhs_term, flags, true);
   
   code_value_void(self);

   conname_next();
   
   return self;
}

CodeNode* i_vabs(CodeNode* self)
{
   Symbol*      sym;
   Term*        term;
   Term const*  term_abs;
   Numb*        rhs;
   unsigned int flags = 0;
   char const*  cname;
   Bound*       lower;
   Bound*       upper;
   Bound const* bigger;
   Bound*       bound_zero;
   Bound*       bound_one;
   Entry*       entry_xplus;
   Entry*       entry_xminus;
   Entry*       entry_bplus;
   Entry*       entry_result;
      
   Trace("i_vabs");
   
   assert(code_is_valid(self));

   term_abs = code_eval_child_term(self, 0);
   rhs      = numb_copy(term_get_constant(term_abs));
   term     = term_copy(term_abs);

   /* Check if trival infeasible
    */
   if (term_get_elements(term) == 0)
   {
      fprintf(stderr, "*** Error 182: Empty LHS, in variable vabs\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (!term_is_all_integer(term))
   {
      fprintf(stderr, "*** Error 183: vabs term not all integer\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   cname      = conname_get();
   bound_zero = bound_new(BOUND_VALUE, numb_zero());
   bound_one  = bound_new(BOUND_VALUE, numb_one());
   lower      = term_get_lower_bound(term);
   upper      = term_get_upper_bound(term);

   if (bound_get_type(lower) != BOUND_VALUE || bound_get_type(upper) != BOUND_VALUE)
   {
      fprintf(stderr, "*** Error 184: vabs term not bounded\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   numb_neg(rhs);
   term_add_constant(term, rhs);

   if (numb_cmp(bound_get_value(lower), numb_zero()) < 0)
   {
      Numb* numb = numb_copy(bound_get_value(lower));
      numb_abs(numb);
      bound_free(lower);
      lower = bound_new(BOUND_VALUE, numb);
      numb_free(numb);
   }
   else
   {
      bound_free(lower);
      lower = bound_new(BOUND_VALUE, numb_zero());
   }
   
   if (numb_cmp(bound_get_value(upper), numb_zero()) < 0)
   {
      bound_free(upper);
      upper = bound_new(BOUND_VALUE, numb_zero());
   }

   bigger = (numb_cmp(bound_get_value(lower), bound_get_value(upper)) > 0)
      ? lower : upper;
   
   /* Create x^+, x^-, Result
    */
   entry_xplus  = create_new_var_entry(cname, "_xp", VAR_INT, bound_zero, upper);
   entry_xminus = create_new_var_entry(cname, "_xm", VAR_INT, bound_zero, lower);
   entry_bplus  = create_new_var_entry(cname, "_bp", VAR_INT, bound_zero, bound_one);
   entry_result = create_new_var_entry(cname, "_re", VAR_INT, bound_zero, bigger);

   /* add term = x^+ + x^-
    */
   term_add_elem(term, entry_xplus, numb_minusone(), MFUN_NONE);
   term_add_elem(term, entry_xminus, numb_one(), MFUN_NONE);
   create_new_constraint(cname, "_a", term, CON_EQUAL, rhs, flags);
   
   /* bplus * upper >= xplus */
   term = term_new(2);
   if (!numb_equal(bound_get_value(upper), numb_zero()))
      term_add_elem(term, entry_bplus, bound_get_value(upper), MFUN_NONE);
   term_add_elem(term, entry_xplus, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_b", term, CON_LHS, numb_zero(), flags);

   /* (1 - bplus) * lower >= xminus
    * lower - bplus * lower - xminus >= 0
    * - bplus * lower - xminus >= -lower
    * xminus + bplus * lower <= lower
    */
   term = term_new(2);
   if (!numb_equal(bound_get_value(lower), numb_zero()))
      term_add_elem(term, entry_bplus, bound_get_value(lower), MFUN_NONE);
   term_add_elem(term, entry_xminus, numb_one(), MFUN_NONE);
   create_new_constraint(cname, "_c", term, CON_RHS, bound_get_value(lower), flags);

   /* result - xplus - xminus == 0
    */
   term = term_new(3);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);
   term_add_elem(term, entry_xplus,  numb_minusone(), MFUN_NONE);
   term_add_elem(term, entry_xminus, numb_minusone(), MFUN_NONE);
   create_new_constraint(cname, "_d", term, CON_EQUAL, numb_zero(), flags);

   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   /* Symbol for internal entries
    */
   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_xplus);
   symbol_add_entry(sym, entry_xminus);
   symbol_add_entry(sym, entry_bplus);
   symbol_add_entry(sym, entry_result);
   
   bound_free(bound_one);
   bound_free(bound_zero);
   bound_free(lower);
   bound_free(upper);
   numb_free(rhs);
   
   return self;
}

CodeNode* i_vexpr_fun(CodeNode* self)
{
   Symbol*      sym;
   Term const*  term_fun;
   char const*  cname;
   Numb const*  expo;
   Numb const*  funno;
   unsigned int flags = 0;
   Term*        term;
   Entry*       entry_tmp;
   Entry*       entry_result;
   Bound*       t_lower;
   Bound*       t_upper;
   Bound*       r_lower;
   Bound*       r_upper;
   MFun         mfun;
   Numb*        rhs;
   
   Trace("i_vexpr_fun");

   /* r = result, t = temp
    *
    * r = f(vexpr)
    * t = vexpr  => vexpr - t == 0
    * r = f(t)   => f(t)  - r == 0
    *
    * WARNING: The coefficent of a monom that is a function
    * is used as the exponent for functions that need a second
    * number argument, e.g. pow(x*y, 3)
    */
   assert(code_is_valid(self));

   funno = code_eval_child_numb(self, 0);

   assert(numb_is_int(funno));
   
   //mfun         = (MFun)numb_toint(funno);
   mfun         = numb_toint(funno);
   term_fun     = code_eval_child_term(self, 1);
   
   switch(mfun)
   {
   case MFUN_SQRT : 
      t_lower = bound_new(BOUND_VALUE, numb_zero());
      t_upper = bound_new(BOUND_INFTY, NULL);
      r_lower = bound_new(BOUND_VALUE, numb_zero());
      r_upper = bound_new(BOUND_INFTY, NULL);
      break;
   case MFUN_LOG :
   case MFUN_LN  :
      t_lower = bound_new(BOUND_VALUE, numb_zero());
      t_upper = bound_new(BOUND_INFTY, NULL);
      r_lower = bound_new(BOUND_MINUS_INFTY, NULL);
      r_upper = bound_new(BOUND_INFTY, NULL);
      break;
   case MFUN_EXP :
   case MFUN_TAN :
   case MFUN_POW :
   case MFUN_SGNPOW :
      t_lower = bound_new(BOUND_MINUS_INFTY, NULL);
      t_upper = bound_new(BOUND_INFTY, NULL);
      r_lower = bound_new(BOUND_MINUS_INFTY, NULL);
      r_upper = bound_new(BOUND_INFTY, NULL);
      break;
   case MFUN_SIN :
   case MFUN_COS :
   case MFUN_SGN :
      t_lower = bound_new(BOUND_MINUS_INFTY, NULL);
      t_upper = bound_new(BOUND_INFTY, NULL);
      r_lower = bound_new(BOUND_VALUE, numb_minusone());
      r_upper = bound_new(BOUND_VALUE, numb_one());
      break;
   case MFUN_ABS :
      t_lower = bound_new(BOUND_MINUS_INFTY, NULL);
      t_upper = bound_new(BOUND_INFTY, NULL);
      r_lower = bound_new(BOUND_VALUE, numb_zero());
      r_upper = bound_new(BOUND_INFTY, NULL);
      break;
   default :
      abort();
   }

   if (mfun == MFUN_POW || mfun == MFUN_SGNPOW)
      expo = code_eval_child_numb(self, 2);
   else
      expo = numb_one();
   
   cname        = conname_get();
   entry_tmp    = create_new_var_entry(cname, "_t", VAR_CON, t_lower, t_upper);
   entry_result = create_new_var_entry(cname, "_r", mfun == MFUN_SGN ? VAR_INT : VAR_CON, r_lower, r_upper);

   /* vexpr - t == 0
    */
   term         = term_copy(term_fun);
   term_add_elem(term, entry_tmp, numb_minusone(), MFUN_NONE); 

   rhs = numb_new_mul(term_get_constant(term), numb_minusone());
   term_add_constant(term, rhs);

   create_new_constraint(cname, "_a", term, CON_EQUAL, rhs, flags);

   numb_free(rhs);

   /* fun(t) - r == 0
    */
   term = term_new(2);
   term_add_elem(term, entry_result, numb_minusone(), MFUN_NONE);
   term_add_elem(term, entry_tmp, expo, mfun);
   
   create_new_constraint(cname, "_b", term, CON_EQUAL, numb_zero(), flags);
  
   term = term_new(1);
   term_add_elem(term, entry_result, numb_one(), MFUN_NONE);

   code_value_term(self, term);

   sym  = symbol_lookup(SYMBOL_NAME_INTERNAL);

   assert(sym != NULL);
   
   symbol_add_entry(sym, entry_tmp);
   symbol_add_entry(sym, entry_result);
   
   bound_free(t_lower);
   bound_free(t_upper);
   bound_free(r_lower);
   bound_free(r_upper);
   
   return self;
}

  
expects_NONNULL
void addcon_as_qubo(
   CodeNode const* const self,   
   ConType         const contype,    /**< Type of constraint (LHS, RHS, EQUAL, RANGE, etc) */
   Numb     const* const rhs,        /**< term contype rhs.*/
   Term     const* const term_org,   /**< term to use */
   unsigned int    const flags)
{
   assert(rhs  != NULL);
   assert(term_is_valid(term_org));
   assert(numb_equal(term_get_constant(term_org), numb_zero()));

   Term* const term  = term_simplify(term_org);
   Bound*      bound = NULL;
   
   if (!term_is_linear(term))
   {
      fprintf(stderr, "*** Error 403: Non linear term can't be converted to QUBO\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   // ax = b => ax - b = 0
   term_sub_constant(term, rhs);

   switch(contype)
   {
   case CON_EQUAL : /* In case of EQUAL, both should be equal */
      bound = bound_new(BOUND_VALUE, numb_zero());
      break;
   case CON_RHS   :
      bound = term_get_lower_bound(term);
      break;
   case CON_LHS :
      bound = term_get_lower_bound(term);
      break;
   case CON_RANGE :
      fprintf(stderr, "*** Error 402: Less equal, greater equal and range can't be converted to QUBO (yet)\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   // TODO: Check that bound_get_type(bound) == BOUND_VALUE otherwise error

   // Do we need to add slack?
   if (contype != CON_EQUAL)
   {
      Numb const* const slack      = bound_get_value(bound);

      assert(numb_is_int(slack));
   
      int         const sval       = abs(numb_toint(slack));

      assert(sval > 0);

      // fprintf(stderr, "sval = %d\n", sval);
      
      Bound*      const bound_zero = bound_new(BOUND_VALUE, numb_zero());
      Bound*      const bound_one  = bound_new(BOUND_VALUE, numb_one());
      char const* const cname      = conname_get();
      Symbol*     const sym        = symbol_lookup(SYMBOL_NAME_INTERNAL);

      assert(sym != NULL);

      // if (sval > 1 << 30)
      //   error too big;
      
      /* binäre zerlegung:
          1 => [] 1
          2 => 1 [] 1
          3 => 1 2
          4 => 1 2 [] 1
          5 => 1 2 [] 2
          6 => 1 2 [] 1 2
          7 => 1 2 4
          8 => 1 2 4 [] 1
          9 => 1 2 4 [] 2
         10 => 1 2 4 [] 1 2
         11 => 1 2 4 [] 4
         12 => 1 2 4 [] 1 4
         13 => 1 2 4 [] 2 4
         14 => 1 2 4 [] 1 2 4
         15 => 1 2 4 8 [] 
         15 => 1 2 4 8 [] 1
         17 => 1 2 4 8 [] 2
      */
      int p2 = 0;

      for(; sval > (1 << (p2 + 1)); p2++)
      {
         Entry* const entry_slack = create_new_var_entry(cname, "_sl", VAR_INT, bound_zero, bound_one);      
         Numb*  const coeff_slack = numb_new_integer((1 << p2) * ((contype == CON_RHS) ? 1 : -1));

         // fprintf(stderr, "added p2 = %d %d\n", p2, (1 << p2) * ((contype == CON_RHS) ? 1 : -1));
         
         symbol_add_entry(sym, entry_slack);
         term_add_elem(term, entry_slack, coeff_slack, MFUN_NONE);

         numb_free(coeff_slack);
      }
      int rest = sval - ((1 << p2) - 1);

      // fprintf(stderr, "rest= %d  p2=%d\n", rest, p2);
      
      for(int i = 0; i < 30; i++)
      {
         if ((rest & (1 << i)) == 0)
            continue;

         Entry* const entry_slack = create_new_var_entry(cname, "_sl", VAR_INT, bound_zero, bound_one);      
         Numb*  const coeff_slack = numb_new_integer((1 << i) * ((contype == CON_RHS) ? 1 : -1));
         
         // fprintf(stderr, "added i = %d %d\n", i, (1 << i) * ((contype == CON_RHS) ? 1 : -1));

         symbol_add_entry(sym, entry_slack);
         term_add_elem(term, entry_slack, coeff_slack, MFUN_NONE);

         numb_free(coeff_slack);
      }
      bound_free(bound_zero);
      bound_free(bound_one);
   }
   bound_free(bound);

   // P * (Ax - b)^2
   // TODO: Check integral coefficients!

   Term* qterm   = term_mul_term(term, term);
   int   penalty = 1;
   
   if      (flags & LP_FLAG_CON_PENALTY1)
      penalty = 10;
   else if (flags & LP_FLAG_CON_PENALTY2)
      penalty = 100;
   else if (flags & LP_FLAG_CON_PENALTY3)
      penalty = 1000;
   else if (flags & LP_FLAG_CON_PENALTY4)
      penalty = 10000;
   else if (flags & LP_FLAG_CON_PENALTY5)
      penalty = 100000;
   else if (flags & LP_FLAG_CON_PENALTY6)
      penalty = 1000000;
   
   Numb* const penalty_factor = numb_new_integer(penalty);
   
   term_mul_coeff(qterm, penalty_factor);

   // term_print(stderr, qterm, false);
   
   xlp_addtoobj(prog_get_lp(), qterm);

   numb_free(penalty_factor);   
   term_free(qterm);
   term_free(term);
}





