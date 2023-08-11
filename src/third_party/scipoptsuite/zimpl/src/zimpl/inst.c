/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: inst.c                                                        */
/*   Name....: Instructions                                                  */
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

/* #define TRACE  1 */

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
#include "zimpl/local.h"
#include "zimpl/list.h"
#include "zimpl/entry.h"
#include "zimpl/heap.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"
#include "zimpl/prog.h"
#include "zimpl/xlpglue.h"
#include "zimpl/strstore.h"

static int checked_eval_numb_toint(CodeNode const* self, int no, char const* errmsg)
{
   assert(self   != NULL);
   assert(no     >= 0);
   assert(errmsg != NULL);

   Numb const* const numb = code_eval_child_numb(self, no);
   
   if (!numb_is_int(numb))
   {
      fprintf(stderr, "*** Error %s ", errmsg);
      numb_print(stderr, numb);
      fprintf(stderr, " is too big or not an integer\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   return numb_toint(numb);
}
/* ----------------------------------------------------------------------------
 * Kontrollfluss Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_nop(CodeNode* self)
{
   Trace("i_nop");

   assert(code_is_valid(self));

   if (code_get_type(self) == CODE_ERR)
      code_value_void(self);

   return self;
}

CodeNode* i_subto(CodeNode* self)
{
   Trace("i_subto");
   
   assert(code_is_valid(self));

   char const* const name = code_eval_child_name(self, 0);

   if (!conname_set(name))
   {
      fprintf(stderr, "*** Error 105: Duplicate constraint name \"%s\"\n", name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   (void)code_eval_child(self, 1); /* constraint */

   conname_free();
   
   code_value_void(self);

   return self;
}

CodeNode* i_constraint_list(CodeNode* self)
{
   Trace("i_constraint_list");
   
   assert(code_is_valid(self));

   (void)code_eval_child(self, 0); /* constraint */
   (void)code_eval_child(self, 1); /* constraint */

   code_value_void(self);

   return self;
}

#if 0
expects_NONNULL
static void addcon_as_qubo(
   CodeNode const* const self,   
   ConType         const contype,    /**< Type of constraint (LHS, RHS, EQUAL, RANGE, etc) */
   Numb     const* const rhs,        /**< term contype rhs.*/
   Term     const* const term_org,   /**< term to use */
   unsigned int    const flags)
{
   assert(rhs  != NULL);
   assert(term_is_valid(term_org));
   assert(numb_equal(term_get_constant(term_org), numb_zero()));

   switch(contype)
   {
   case CON_EQUAL : /* In case of EQUAL, both should be equal */
      if (!numb_equal(rhs, numb_one()))
      {
         fprintf(stderr, "*** Error 401: RHS unequal to 1 can't be converted to QUBO (yet)\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      break;
   case CON_RHS   :
   case CON_LHS :
      //if (!nump_equal(lhs, numb_one))
      fall_THROUGH
   case CON_RANGE :
      fprintf(stderr, "*** Error 402: Less equal, greater equal and range can't be converted to QUBO (yet)\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   Term* const term = term_simplify(term_org);

   if (!term_is_linear(term))
   {
      fprintf(stderr, "*** Error 403: Non linear term can't be converted to QUBO\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }

   /*
      x             == 1 => P(1 -x)
      x + y         == 1 => P(1 -x -y +2xy)
      x + y + z     == 1 => P(1 -x -y -z +2xy +2xz +2yz)
      a + b + c + d == 1 => P(1 -a -b -c -d +2ab +2ac +2ad +2bc +2bd +2cd)
      
      x         >= 1 => P(1 -x)  # x can't be > 1
      x + y     >= 1 => P(1 -x -y +xy)
      Wrong:
      x + y + z >= 1 => P(1 -x -y -z +xy +xz +yz)
      a + b + c + d >= 1 => P(1 -a -b -c -d +ab +ac +ad +bc +bd +cd)

      General:
      i = 1..n
      p = n!/(2*(n-2)!)

      if a of the x_i = 1
      => a!/(2*(a-2)!) of the pairs = 1

      a  A     B    Offset 
      1  1     2      1
      2  3/2   1      2
      3  4/3  1/3     3
      4  5/4  1/6     4
      5  6/5  1/10    5

      ab 3
      A = (a + 1)/a
      B = 1/(a!/(2*(a-2)!))= 2*(a-2)!/a! = 2/(a * (a - 1))
      sum x_i == a => P(  - A x_i + B x_i x_j      
   */
   int   const telems = term_get_elements(term);   
   Term* const qterm  = term_new(telems * telems + telems);

   term_add_constant(qterm, numb_one());

   for(int i = 0; i < telems; i++)
   {
      Mono const* mono1 = term_get_element(term, i);
   
      assert(mono_is_linear(mono1));

      if (mono_get_function(mono1) != MFUN_NONE)
      {
         //??? not sure this can be reached
         fprintf(stderr, "*** Error 404: Non linear expressions can't be converted to QUBO\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      if (!numb_equal(mono_get_coeff(mono1), numb_one()))
      {
         fprintf(stderr, "*** Error 405: Constraints with coefficients != 1 can't be converted to QUBO\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }    
      for(int k = 0; k < telems; k++)
      {
         Mono const* mono2 = term_get_element(term, k);
         Mono*       mono  = mono_mul(mono1, mono2);
         
         if (i == k)
            mono_neg(mono);

         term_append_elem(qterm, mono);
      }
   }
   int penalty = 1;
   
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
   //Numb* const penalty_factor = numb_new_integer(1);
   
   term_mul_coeff(qterm, penalty_factor);

   xlp_addtoobj(prog_get_lp(), qterm);

   numb_free(penalty_factor);   
   term_free(qterm);
   term_free(term);
}
#endif

CodeNode* i_constraint(CodeNode* self)
{
   Trace("i_constraint");
   
   assert(code_is_valid(self));

   Term const*  const term_lhs = code_eval_child_term(self, 0);
   ConType      const type     = code_eval_child_contype(self, 1);
   Term const*  const term_rhs = code_eval_child_term(self, 2);
   unsigned int const flags    = code_eval_child_bits(self, 3);

   Numb*        const rhs      = numb_new_sub(term_get_constant(term_rhs), term_get_constant(term_lhs));
   Term*        const term     = term_sub_term(term_lhs, term_rhs);

   /* Check if trival infeasible
    */
   if (term_get_elements(term) == 0)
   {
      /* If zero, trival ok, otherwise ...
       */
      int res = numb_cmp(rhs, numb_zero());

      assert(type != CON_RANGE);
      assert(type != CON_FREE);
      
      if (  (type == CON_EQUAL && res != 0)
         || (type == CON_LHS   && res >  0)
         || (type == CON_RHS   && res <  0))
      {
         fprintf(stderr, "*** Error 106: Empty LHS, constraint trivially violated\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   else
   {
      term_add_constant(term, rhs);

      if (flags & LP_FLAG_CON_QUBO)
      {
         addcon_as_qubo(self, type, rhs, term, flags);
      }
      else
      {      
         if (xlp_addcon_term(prog_get_lp(), conname_get(), type, rhs, rhs, flags, term))
            code_errmsg(self);
      }
      conname_next();
   }
   code_value_void(self);

   numb_free(rhs);
   term_free(term);
   
   return self;
}

CodeNode* i_rangeconst(CodeNode* self)
{
   Trace("i_rangeconst");
   
   assert(code_is_valid(self));
   
   /* It has to be either l <= x <= u, or u >= x >= l
    */
   if (code_eval_child_contype(self, 3) != code_eval_child_contype(self, 4))
   {
      fprintf(stderr, "*** Error 107: Range must be l <= x <= u, or u >= x >= l\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   
   Numb*        const lhs   = numb_copy(code_eval_child_numb(self, 0));
   Term*        const term  = term_copy(code_eval_child_term(self, 1));
   Numb*        const rhs   = numb_copy(code_eval_child_numb(self, 2));
   unsigned int const flags = code_eval_child_bits(self, 5);

   numb_sub(lhs, term_get_constant(term));
   numb_sub(rhs, term_get_constant(term));
      
   /* Check if trival infeasible
    */
   if (term_get_elements(term) == 0)
   {
      /* If zero, trival ok, otherwise ...
       */
      if (numb_cmp(lhs, numb_zero()) > 0 || numb_cmp(rhs, numb_zero()) < 0)
      {
         fprintf(stderr,
            "*** Error 108: Empty Term with nonempty LHS/RHS, constraint trivially violated\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   else
   {
      if (numb_cmp(lhs, rhs) > 0)
      {
         fprintf(stderr, "*** Error 109: LHS/RHS contradiction, constraint trivially violated\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      term_sub_constant(term, term_get_constant(term));

      if (xlp_addcon_term(prog_get_lp(), conname_get(), CON_RANGE, lhs, rhs, flags, term))
         code_errmsg(self);

      conname_next();
   }
   code_value_void(self);

   numb_free(rhs);
   numb_free(lhs);
   term_free(term);
   
   return self;
}

CodeNode* i_sos(CodeNode* self)
{
   Trace("i_sos");
   
   assert(code_is_valid(self));

   char const* const name = code_eval_child_name(self, 0);

   if (!conname_set(name))
   {
      fprintf(stderr, "*** Error 105: Duplicate constraint name \"%s\"\n", name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   (void)code_eval_child(self, 1); /* soset */

   conname_free();
   
   code_value_void(self);

   return self;
}

CodeNode* i_soset(CodeNode* self)
{
   Trace("i_soset");
   
   assert(code_is_valid(self));

   Term const* const term       = code_eval_child_term(self, 0);
   Numb const* const typenumb   = code_eval_child_numb(self, 1);
   Numb const* const priority   = code_eval_child_numb(self, 2);

   if (!numb_equal(term_get_constant(term), numb_zero()))
   {
      fprintf(stderr, "*** Error 199: Constants are not allowed in SOS declarations\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   SosType const type = (numb_equal(typenumb, numb_one())) ? SOS_TYPE1 : SOS_TYPE2;
   
   int ret = xlp_addsos_term(prog_get_lp(), conname_get(), type, priority, term);

   if ((ret & 1) && stmt_trigger_warning(200))
   {
      fprintf(stderr,
         "--- Warning 200: Weights are not unique for SOS %s\n", conname_get());
      code_errmsg(self);
   }
   if ((ret & 2) && stmt_trigger_warning(302))
   {
      fprintf(stderr,
         "--- Warning 302: Priority has to be integral for SOS %s\n", conname_get());
      code_errmsg(self);
   }  
   conname_next();

   code_value_void(self);

   return self;
}

static bool has_pattern_name(
   Tuple const* pattern)
{
   int dim = tuple_get_dim(pattern);

   int i;
   for(i = 0; i < dim; i++)
      if (ELEM_NAME == elem_get_type(tuple_get_elem(pattern, i)))
         break;

   return i < dim;
}

static void warn_if_pattern_has_no_name(
   CodeNode const* self,
   Tuple const*    pattern)
{
   if (tuple_get_dim(pattern) > 0 && !has_pattern_name(pattern))
   {
      if (stmt_trigger_warning(203))
      {
         fprintf(stderr, "--- Warning 203: Indexing tuple is fixed\n");
         code_errmsg(self);
      }
   }
}

CodeNode* i_forall(CodeNode* self)
{
   Trace("i_forall");
   
   assert(code_is_valid(self));

   IdxSet const* const idxset  = code_eval_child_idxset(self, 0);  
   Set    const* const set     = idxset_get_set(idxset);
   Tuple  const* const pattern = idxset_get_tuple(idxset);
   CodeNode*     const lexpr   = idxset_get_lexpr(idxset);
   SetIter*      const iter    = set_iter_init(set, pattern);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);
   
   Tuple* tuple;

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
         (void)code_eval_child(self, 1); /* z.B. constraint */

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);
   
   code_value_void(self);

   return self;
}

/* ----------------------------------------------------------------------------
 * Arithmetische Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_expr_add(CodeNode* self)
{
   Trace("i_expr_add");

   assert(code_is_valid(self));

   CodeNode* const child = code_eval_child(self, 0);

   if (code_get_type(child) == CODE_NUMB)
      code_value_numb(self, numb_new_add(code_get_numb(child), code_eval_child_numb(self, 1)));
   else
   {
      char const* s1 = code_get_strg(child);
      char const* s2 = code_eval_child_strg(self, 1);
      char* t        = malloc(strlen(s1) + strlen(s2) + 1);

      assert(t != NULL);

      strcpy(t, s1);
      strcat(t, s2);

      code_value_strg(self, str_new(t));

      free(t);
   }
   return self;
}

CodeNode* i_expr_sub(CodeNode* self)
{
   Trace("i_expr_sub");

   assert(code_is_valid(self));

   code_value_numb(self,
      numb_new_sub(code_eval_child_numb(self, 0), code_eval_child_numb(self, 1)));

   return self;
}

CodeNode* i_expr_mul(CodeNode* self)
{
   Trace("i_expr_mul");

   assert(code_is_valid(self));

   code_value_numb(self,
      numb_new_mul(code_eval_child_numb(self, 0), code_eval_child_numb(self, 1)));

   return self;
}

CodeNode* i_expr_div(CodeNode* self)
{
   Trace("i_expr_div");

   assert(code_is_valid(self));

   Numb const* const divisor = code_eval_child_numb(self, 1);

   if (numb_equal(divisor, numb_zero()))
   {
      fprintf(stderr, "*** Error 110: Division by zero\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self,
      numb_new_div(code_eval_child_numb(self, 0), divisor));

   return self;
}

CodeNode* i_expr_mod(CodeNode* self)
{
   Trace("i_expr_mod");

   assert(code_is_valid(self));

   Numb const* const divisor = code_eval_child_numb(self, 1);

   if (numb_equal(divisor, numb_zero()))
   {
      fprintf(stderr, "*** Error 111: Modulo by zero\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self,
      numb_new_mod(code_eval_child_numb(self, 0), divisor));

   return self;
}

CodeNode* i_expr_intdiv(CodeNode* self)
{
   Trace("i_expr_intdiv");

   assert(code_is_valid(self));

   Numb const* const divisor = code_eval_child_numb(self, 1);

   if (numb_equal(divisor, numb_zero()))
   {
      fprintf(stderr, "*** Error 110: Division by zero\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self,
      numb_new_intdiv(
         code_eval_child_numb(self, 0), divisor));

   return self;
}

CodeNode* i_expr_pow(CodeNode* self)
{
   Trace("i_expr_pow");

   assert(code_is_valid(self));

   int ex = checked_eval_numb_toint(self, 1, "112: Exponent value");
   
   code_value_numb(self,
      numb_new_pow(code_eval_child_numb(self, 0), ex));

   return self;
}

CodeNode* i_expr_neg(CodeNode* self)
{
   Trace("i_expr_neg");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));

   numb_neg(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_abs(CodeNode* self)
{
   Trace("i_expr_abs");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));

   numb_abs(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_sgn(CodeNode* self)
{
   Trace("i_expr_sgn");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));

   numb_sgn(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_floor(CodeNode* self)
{
   Trace("i_expr_floor");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));

   numb_floor(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_ceil(CodeNode* self)
{
   Trace("i_expr_ceil");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));
   
   numb_ceil(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_log(CodeNode* self)
{
   Trace("i_expr_log");

   assert(code_is_valid(self));

   Numb* const numb = numb_new_log(code_eval_child_numb(self, 0));
   
   if (numb == NULL)
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_ln(CodeNode* self)
{
   Trace("i_expr_ln");

   assert(code_is_valid(self));

   Numb* const numb = numb_new_ln(code_eval_child_numb(self, 0));
   
   if (numb == NULL)
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_sqrt(CodeNode* self)
{
   Trace("i_expr_sqrt");

   assert(code_is_valid(self));

   Numb* const numb = numb_new_sqrt(code_eval_child_numb(self, 0));
   
   if (numb == NULL)
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_exp(CodeNode* self)
{
   Trace("i_expr_exp");

   assert(code_is_valid(self));

   code_value_numb(self, numb_new_exp(code_eval_child_numb(self, 0)));
   
   return self;
}

CodeNode* i_expr_sin(CodeNode* self)
{
   Trace("i_expr_sin");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_cos(CodeNode* self)
{
   Trace("i_expr_cos");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_tan(CodeNode* self)
{
   Trace("i_expr_tan");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_asin(CodeNode* self)
{
   Trace("i_expr_asin");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_acos(CodeNode* self)
{
   Trace("i_expr_acos");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_atan(CodeNode* self)
{
   Trace("i_expr_atan");

   assert(code_is_valid(self));

   fprintf(stderr, "Not implemented yet\n");

   /* ??? numb = numb_new_log(code_eval_child_numb(self, 0)); */
   
   Numb* numb = NULL;

   if (numb == NULL) //lint !e774 conditionalways true
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }      
   code_value_numb(self, numb);
   
   return self;
}

CodeNode* i_expr_fac(CodeNode* self)
{
   Trace("i_expr_fac");

   assert(code_is_valid(self));

   int n = checked_eval_numb_toint(self, 0, "113: Factorial value");
   
   if (n < 0)
   {
      fprintf(stderr, "*** Error 114: Negative factorial value\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (n > 1000)
   {
      fprintf(stderr, "*** Error 115: Timeout!\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   code_value_numb(self, numb_new_fac(n));

   return self;
}

CodeNode* i_expr_card(CodeNode* self)
{
   Trace("i_card");

   assert(code_is_valid(self));

   Set const* const set = code_eval_child_set(self, 0);

   code_value_numb(self, numb_new_integer(set_get_members(set)));

   return self;
}

CodeNode* i_expr_rand(CodeNode* self)
{
   Trace("i_rand");

   assert(code_is_valid(self));

   Numb const* const mini = code_eval_child_numb(self, 0);
   Numb const* const maxi = code_eval_child_numb(self, 1);

   if (numb_cmp(mini, maxi) >= 0)
   {
      fprintf(stderr, "*** Error 204: Randomfunction parameter minimum= ");
      numb_print(stderr, mini);
      fprintf(stderr, " >= maximum= ");
      numb_print(stderr, maxi);
      fprintf(stderr, "\n");
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   }
   code_value_numb(self, numb_new_rand(mini, maxi));
   
   return self;
}

CodeNode* i_expr_round(CodeNode* self)
{
   Trace("i_expr_round");

   assert(code_is_valid(self));

   Numb* const numb = numb_copy(code_eval_child_numb(self, 0));
   
   numb_round(numb);
   
   code_value_numb(self, numb);

   return self;
}

CodeNode* i_expr_if_else(CodeNode* self)
{
   Trace("i_if");

   assert(code_is_valid(self));

   if (code_eval_child_bool(self, 0))
      code_copy_value(self, code_eval_child(self, 1));
   else
      code_copy_value(self, code_eval_child(self, 2));

   return self;
}

CodeNode* i_expr_min(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Numb const*   value;
   Numb*         min   = numb_new();
   bool          first = true;
   
   Trace("i_expr_min");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         value = code_eval_child_numb(self, 1);      

         if (first || numb_cmp(value, min) < 0)
         {
            numb_set(min, value);
            first = false;
         }
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);
   
   if (first)
   {
      if (stmt_trigger_warning(186))
      {
         fprintf(stderr, "--- Warning 186: Minimizing over empty set -- zero assumed\n");
         code_errmsg(code_get_child(self, 0));
      }
   }
   code_value_numb(self, min);

   return self;
}

CodeNode* i_expr_sglmin(CodeNode* self)
{
   Trace("i_expr_sglmin");
   
   assert(code_is_valid(self));

   IdxSet const* idxset  = code_eval_child_idxset(self, 0);
   Set const*    set     = idxset_get_set(idxset);
   Tuple const*  pattern = idxset_get_tuple(idxset);
   CodeNode*     lexpr   = idxset_get_lexpr(idxset);
   Numb*         min     = numb_new();
   bool          first   = true;

   if (set_get_dim(set) != 1)
   {
      fprintf(stderr, "*** Error 209: MIN of set with more than one dimension\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }

   if (set_get_members(set) > 0)
   {
      Tuple* tuple = set_get_tuple(set, 0);

      if (elem_get_type(tuple_get_elem(tuple, 0)) != ELEM_NUMB)
      {
         fprintf(stderr, "*** Error 211: MIN of set containing non number elements\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      tuple_free(tuple);
   
      SetIter* iter = set_iter_init(set, pattern);

      while((tuple = set_iter_next(iter, set)) != NULL)
      {
         local_install_tuple(pattern, tuple);
         
         if (code_get_bool(code_eval(lexpr)))
         {
            Numb const* value = elem_get_numb(tuple_get_elem(tuple, 0));      
            
            if (first || numb_cmp(value, min) < 0)
            {
               numb_set(min, value);
               first = false;
            }
         }
         local_drop_frame();
         
         tuple_free(tuple);
      }
      set_iter_exit(iter, set);
   }
   if (first)
   {
      if (stmt_trigger_warning(186))
      {
         fprintf(stderr,
            "--- Warning 186: Minimizing over empty set -- zero assumed\n");
         code_errmsg(code_get_child(self, 0));
      }
   }
   code_value_numb(self, min);

   return self;
}

CodeNode* i_expr_max(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Numb const*   value;
   Numb*         max   = numb_new();
   bool          first = true;

   Trace("i_expr_max");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);
   
   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         value = code_eval_child_numb(self, 1);      

         if (first || numb_cmp(value, max) > 0)
         {
            numb_set(max, value);
            first = false;
         }
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);
   
   if (first)
   {
      if (stmt_trigger_warning(187))
      {
         fprintf(stderr, "--- Warning 187: Maximizing over empty set -- zero assumed\n");
         code_errmsg(code_get_child(self, 0));
      }
   }
   code_value_numb(self, max);

   return self;
}

CodeNode* i_expr_sglmax(CodeNode* self)
{
   Trace("i_expr_max");
   
   assert(code_is_valid(self));

   IdxSet const* idxset  = code_eval_child_idxset(self, 0);
   Set const*    set     = idxset_get_set(idxset);
   Tuple const*  pattern = idxset_get_tuple(idxset);
   CodeNode*     lexpr   = idxset_get_lexpr(idxset);
   Numb*         max     = numb_new();
   bool          first   = true;

   if (set_get_dim(set) != 1)
   {
      fprintf(stderr, "*** Error 210: MAX of set with more than one dimension\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (set_get_members(set) > 0)
   {
      Tuple* tuple = set_get_tuple(set, 0);

      if (elem_get_type(tuple_get_elem(tuple, 0)) != ELEM_NUMB)
      {
         fprintf(stderr, "*** Error 212: MAX of set containing non number elements\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      tuple_free(tuple);

      SetIter* iter = set_iter_init(set, pattern);

      while((tuple = set_iter_next(iter, set)) != NULL)
      {
         local_install_tuple(pattern, tuple);
         
         if (code_get_bool(code_eval(lexpr)))
         {
            Numb const* value = elem_get_numb(tuple_get_elem(tuple, 0));      
            
            if (first || numb_cmp(value, max) > 0)
            {
               numb_set(max, value);
               first = false;
            }
         }
         local_drop_frame();
         
         tuple_free(tuple);
      }
      set_iter_exit(iter, set);
   }
   if (first)
   {
      if (stmt_trigger_warning(187))
      {
         fprintf(stderr, "--- Warning 187: Maximizing over empty set -- zero assumed\n");
         code_errmsg(code_get_child(self, 0));
      }
   }
   code_value_numb(self, max);

   return self;
}

CodeNode* i_expr_sum(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Numb*         sum = numb_new();

   Trace("i_expr_sum");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
         numb_add(sum, code_eval_child_numb(self, 1));      

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);

   code_value_numb(self, sum);

   return self;
}

CodeNode* i_expr_prod(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Numb*         prod = numb_new_integer(1);

   Trace("i_expr_prod");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
         numb_mul(prod, code_eval_child_numb(self, 1));      

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);

   code_value_numb(self, prod);

   return self;
}

CodeNode* i_expr_min2(CodeNode* self)
{
   List const*   list;
   Numb*         min;
   ListElem*     le    = NULL;
   int           n;
   bool          first = true;
   
   Trace("i_expr_min2");
   
   assert(code_is_valid(self));

   list  = code_eval_child_list(self, 0);
   n     = list_get_elems(list);
   min   = numb_new();

   assert(n > 0);

   while(n-- > 0)
   {
      Elem const* elem = list_get_elem(list, &le);
      Numb const* numb;

      assert(elem != NULL);
      
      /* Are there only number in the selection tuple ?
       */
      if (ELEM_NUMB != elem_get_type(elem))
      {
         fprintf(stderr, "*** Error 116: Illegal value type in min: ");
         elem_print(stderr, elem, true);
         fprintf(stderr, " only numbers are possible\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      numb = elem_get_numb(elem);

      if (first || numb_cmp(min, numb) > 0)
      {
         numb_set(min, numb);
         first = false;
      }
   }
   code_value_numb(self, min);

   return self;
}

CodeNode* i_expr_max2(CodeNode* self)
{
   List const*   list;
   Numb*         max;
   ListElem*     le    = NULL;
   int           n;
   bool          first = true;
   
   Trace("i_expr_max2");
   
   assert(code_is_valid(self));

   list  = code_eval_child_list(self, 0);
   n     = list_get_elems(list);
   max   = numb_new();

   assert(n > 0);
   
   while(n-- > 0)
   {
      Numb const* numb;
      Elem const* elem = list_get_elem(list, &le);

      assert(elem != NULL);
      
      /* Are there only number in the selection tuple ?
       */
      if (ELEM_NUMB != elem_get_type(elem))
      {
         fprintf(stderr, "*** Error 117: Illegal value type in max: ");
         elem_print(stderr, elem, true);
         fprintf(stderr, " only numbers are possible\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      numb = elem_get_numb(elem);

      if (first || numb_cmp(max, numb) < 0)
      {
         numb_set(max, numb);
         first = false;
      }
   }
   code_value_numb(self, max);

   return self;
}

CodeNode* i_expr_ord(CodeNode* self)
{
   Set const*  set;
   Elem const* elem;
   Tuple*      tuple;
   int         tno;
   int         cno;
   
   Trace("i_expr_ord");

   assert(code_is_valid(self));

   set = code_eval_child_set(self, 0);
   tno = checked_eval_numb_toint(self, 1, "189: Tuple number");
   cno = checked_eval_numb_toint(self, 2, "190: Component number");
   
   if (tno < 1 || tno > set_get_members(set))
   {
      fprintf(stderr, "*** Error 191: Tuple number %d", tno);
      fprintf(stderr, " is not a valid value between 1..%d\n", set_get_members(set));
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (cno < 1 || cno > set_get_dim(set))
   {
      fprintf(stderr, "*** Error 192: Component number %d", cno);
      fprintf(stderr, " is not a valid value between 1..%d\n", set_get_dim(set));
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   tuple = set_get_tuple(set, tno - 1);
   elem  = tuple_get_elem(tuple, cno - 1);
   
   switch(elem_get_type(elem))
   {
   case ELEM_NUMB :
      code_value_numb(self, numb_copy(elem_get_numb(elem)));
      break;
   case ELEM_STRG :
      code_value_strg(self, elem_get_strg(elem));
      break;
   default :
      abort();
   }
   tuple_free(tuple);

   return self;
}

CodeNode* i_expr_length(CodeNode* self)
{
   Trace("i_expr_length");

   assert(code_is_valid(self));

   code_value_numb(self,
      numb_new_integer((int)strlen(code_eval_child_strg(self, 0))));

   return self;
}

CodeNode* i_expr_substr(CodeNode* self)
{
   char const* strg;
   int         beg;
   int         len;
   int         maxlen;
   char*       tmp;

   Trace("i_expr_substr");

   assert(code_is_valid(self));

   strg = code_eval_child_strg(self, 0);
   beg  = checked_eval_numb_toint(self, 1, "217: Begin value");
   len  = checked_eval_numb_toint(self, 2, "218: Length value");

   if (len < 0)
   {
      fprintf(stderr, "*** Error 219: Length value %d in substr is negative\n", len);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   tmp = malloc((size_t)len + 1); 

   if (beg < 0)
   {
      beg = (int)strlen(strg) + beg;

      if (beg < 0)
         beg = 0;
   }
   assert(beg >= 0);
   
   maxlen = (int)strlen(strg) - beg;

   if (maxlen < len)
      len = maxlen;

   if (len < 0)
      len = 0;
   else
      strncpy(tmp, &strg[beg], (size_t)len);

   tmp[len] = '\0';

   code_value_strg(self, str_new(tmp));

   free(tmp);

   return self;
}

/* ----------------------------------------------------------------------------
 * Logische Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_bool_true(CodeNode* self)
{
   Trace("i_bool_true");

   assert(code_is_valid(self));
   
   code_value_bool(self, true);

   return self;
}

CodeNode* i_bool_false(CodeNode* self)
{
   Trace("i_bool_false");

   assert(code_is_valid(self));
   
   code_value_bool(self, false);

   return self;
}

CodeNode* i_bool_not(CodeNode* self)
{
   Trace("i_bool_not");

   assert(code_is_valid(self));
   
   code_value_bool(self, !code_eval_child_bool(self, 0));

   return self;
}

CodeNode* i_bool_and(CodeNode* self)
{
   Trace("i_bool_and");

   assert(code_is_valid(self));
   
   code_value_bool(self,
      code_eval_child_bool(self, 0) && code_eval_child_bool(self, 1));

   return self;
}

CodeNode* i_bool_or(CodeNode* self)
{
   Trace("i_bool_or");

   assert(code_is_valid(self));
   
   code_value_bool(self,
      code_eval_child_bool(self, 0) || code_eval_child_bool(self, 1));

   return self;
}

CodeNode* i_bool_xor(CodeNode* self)
{
   bool a;
   bool b;
   
   Trace("i_bool_or");

   assert(code_is_valid(self));

   a = code_eval_child_bool(self, 0);
   b = code_eval_child_bool(self, 1);

   code_value_bool(self, (a && !b) || (b && !a));

   return self;
}

CodeNode* i_bool_eq(CodeNode* self)
{
   CodeNode* op1;
   CodeNode* op2;
   CodeType  tp1;
   CodeType  tp2;
   bool      result;
   
   Trace("i_bool_eq");

   assert(code_is_valid(self));

   op1 = code_eval_child(self, 0);
   op2 = code_eval_child(self, 1);
   tp1 = code_get_type(op1);
   tp2 = code_get_type(op2);

   if (tp1 != tp2)
   {
      fprintf(stderr, "*** Error 118: Comparison of different types\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(tp1 == tp2);

   switch(tp1)
   {
   case CODE_NUMB :
      result = numb_equal(code_get_numb(op1), code_get_numb(op2));
      break;
   case CODE_STRG :
      result = strcmp(code_get_strg(op1), code_get_strg(op2)) == 0;
      break;
   case CODE_NAME :
      fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
         code_get_name(op1));
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   code_value_bool(self, result);

   return self;
}

CodeNode* i_bool_ne(CodeNode* self)
{
   Trace("i_bool_ne");

   assert(code_is_valid(self));

   code_value_bool(self, !code_get_bool(i_bool_eq(self)));

   return self;
}

CodeNode* i_bool_ge(CodeNode* self)
{
   CodeNode* op1;
   CodeNode* op2;
   CodeType  tp1;
   CodeType  tp2;
   bool      result;
   
   Trace("i_bool_ge");

   assert(code_is_valid(self));

   op1 = code_eval_child(self, 0);
   op2 = code_eval_child(self, 1);
   tp1 = code_get_type(op1);
   tp2 = code_get_type(op2);

   if (tp1 != tp2)
   {
      fprintf(stderr, "*** Error 118: Comparison of different types\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(tp1 == tp2);

   switch(tp1)
   {
   case CODE_NUMB :
      result = numb_cmp(code_get_numb(op1), code_get_numb(op2)) >= 0;
      break;
   case CODE_STRG :
      result = strcmp(code_get_strg(op1), code_get_strg(op2)) >= 0;
      break;
   case CODE_NAME :
      fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
         code_get_name(op1));
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   code_value_bool(self, result);

   return self;
}

CodeNode* i_bool_gt(CodeNode* self)
{
   CodeNode* op1;
   CodeNode* op2;
   CodeType  tp1;
   CodeType  tp2;
   bool      result;
   
   Trace("i_bool_gt");

   assert(code_is_valid(self));

   op1 = code_eval_child(self, 0);
   op2 = code_eval_child(self, 1);
   tp1 = code_get_type(op1);
   tp2 = code_get_type(op2);

   if (tp1 != tp2)
   {
      fprintf(stderr, "*** Error 118: Comparison of different types\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(tp1 == tp2);

   switch(tp1)
   {
   case CODE_NUMB :
      result = numb_cmp(code_get_numb(op1), code_get_numb(op2)) > 0;
      break;
   case CODE_STRG :
      result = strcmp(code_get_strg(op1), code_get_strg(op2)) > 0;
      break;
   case CODE_NAME :
      fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
         code_get_name(op1));
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   code_value_bool(self, result);

   return self;
}

CodeNode* i_bool_le(CodeNode* self)
{
   Trace("i_bool_le");

   assert(code_is_valid(self));
   
   code_value_bool(self, !code_get_bool(i_bool_gt(self)));

   return self;
}

CodeNode* i_bool_lt(CodeNode* self)
{
   Trace("i_bool_lt");

   assert(code_is_valid(self));
   
   code_value_bool(self, !code_get_bool(i_bool_ge(self)));

   return self;
}

CodeNode* i_bool_seq(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;
      
   Trace("i_bool_seq");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);
   
   code_value_bool(self, set_is_equal(set_a, set_b));

   return self;
}

CodeNode* i_bool_sneq(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;

   Trace("i_bool_sneq");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);
   
   code_value_bool(self, !set_is_equal(set_a, set_b));

   return self;
}

CodeNode* i_bool_subs(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;

   Trace("i_bool_subs");

   assert(code_is_valid(self));
   
   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   code_value_bool(self, set_is_subset(set_a, set_b));

   return self;
}

CodeNode* i_bool_sseq(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;

   Trace("i_bool_sseq");

   assert(code_is_valid(self));
   
   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   code_value_bool(self, set_is_subseteq(set_a, set_b));

   return self;
}

static void check_tuple_set_compatible(
   CodeNode const* self,
   Tuple const*    tuple_a,
   Set const*      set_b)
{
   Tuple*      tuple_b;
   int         i;
   int         dim;
   
   /* An empty set is compatible with any tuple.
    */
   if (set_get_members(set_b) == 0)
      return;
   
   dim = set_get_dim(set_b);

   if (tuple_get_dim(tuple_a) != dim)
   {
      fprintf(stderr, "*** Error 188: Index tuple ");
      tuple_print(stderr, tuple_a);
      fprintf(stderr, " has wrong dimension %d, expected %d\n",
         tuple_get_dim(tuple_a),
         dim);

      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   tuple_b = set_get_tuple(set_b, 0);

   assert(tuple_get_dim(tuple_a) == tuple_get_dim(tuple_b));

   for(i = 0; i < tuple_get_dim(tuple_a); i++)
   {
      ElemType elem_type_a = elem_get_type(tuple_get_elem(tuple_a, i));
      ElemType elem_type_b = elem_get_type(tuple_get_elem(tuple_b, i));

      assert(elem_type_b == ELEM_NUMB || elem_type_b == ELEM_STRG);

      if (elem_type_a != elem_type_b)
      {
         fprintf(stderr, "*** Error 198: Incompatible index tuple\nTuple ");
         tuple_print(stderr, tuple_a);
         fprintf(stderr, " component %d is not compatible with ", i + 1);
         tuple_print(stderr, tuple_b);
         fprintf(stderr, "\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   tuple_free(tuple_b);
}

CodeNode* i_bool_is_elem(CodeNode* self)
{
   Tuple const* tuple;
   Set const*   set;
   
   Trace("i_bool_is_elem");

   assert(code_is_valid(self));

   tuple = code_eval_child_tuple(self, 0);
   set   = code_eval_child_set(self, 1);

   check_tuple_set_compatible(self, tuple, set);
   
   code_value_bool(self, set_lookup(set, tuple));

   return self;
}

CodeNode* i_bool_exists(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   bool          exists = false;

   Trace("i_bool_exists");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while(!exists && (tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      exists = code_get_bool(code_eval(lexpr));

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);

   code_value_bool(self, exists);

   return self;
}

/* ----------------------------------------------------------------------------
 * Set Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_set_new_tuple(CodeNode* self)
{
   List const*  list;
   Tuple const* tuple;
   ListElem*    le    = NULL;
   int          dim;
   
   Trace("i_set_new_tuple");
   
   assert(code_is_valid(self));

   list  = code_eval_child_list(self, 0);

   if (!list_is_tuplelist(list))
   {
      /* This errors occurs, if a stream "n+" instead of "<n+>" is used in the template
       * for a "read" statement.
       */
      assert(list_is_entrylist(list));

      fprintf(stderr, "*** Error 214: Wrong type of set elements -- wrong read template?\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }

   tuple = list_get_tuple(list, &le);

   assert(tuple != NULL);

   dim   = tuple_get_dim(tuple);

   /* Is this a empty list with just a dummy argument? 
    */
   if (dim == 0 && list_get_elems(list) == 1)
   {      
      code_value_set(self, set_empty_new(0));
   }
   else
   {   
      le    = NULL; /* Start again */
      
      while(NULL != (tuple = list_get_tuple(list, &le)))
      {
         if (tuple_get_dim(tuple) != dim)
         {
            le = NULL;
            fprintf(stderr, "*** Error 193: Different dimension tuples in set initialisation\n");
            tuple_print(stderr, tuple);
            fprintf(stderr, " vs. ");
            tuple_print(stderr, list_get_tuple(list, &le));
            fprintf(stderr, "\n");
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         for(int i = 0; i < dim; i++)
         {
            ElemType elem_type = elem_get_type(tuple_get_elem(tuple, i));
         
            if (elem_type != ELEM_NUMB && elem_type != ELEM_STRG)
            {
               assert(elem_type == ELEM_NAME);
            
               fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
                  elem_get_name(tuple_get_elem(tuple, i)));
               code_errmsg(self);
               zpl_exit(EXIT_FAILURE);
            }
         }
      }
      code_value_set(self, set_new_from_list(list, SET_CHECK_WARN));
   }
   return self;
}

CodeNode* i_set_new_elem(CodeNode* self)
{
   List const*  list;
   Elem const*  elem;
   ListElem*    le    = NULL;
   
   Trace("i_set_new_elem");

   assert(code_is_valid(self));

   list = code_eval_child_list(self, 0);

   while(NULL != (elem = list_get_elem(list, &le)))
   {
      ElemType elem_type = elem_get_type(elem);
      
      if (elem_type != ELEM_NUMB && elem_type != ELEM_STRG)
      {
         assert(elem_type == ELEM_NAME);
         
         fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
            elem_get_name(elem));
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   code_value_set(self, set_new_from_list(list, SET_CHECK_WARN));

   return self;
}

CodeNode* i_set_pseudo(CodeNode* self)
{
   Trace("i_set_pseudo");

   assert(code_is_valid(self));

   code_value_set(self, set_pseudo_new());

   return self;
}

CodeNode* i_set_empty(CodeNode* self)
{
   int  dim;
   
   Trace("i_set_empty");

   assert(code_is_valid(self));

   dim = code_eval_child_size(self, 0);

   code_value_set(self, set_empty_new(dim));

   return self;
}

static void check_sets_compatible(
   CodeNode const* self,
   Set const*      set_a,
   Set const*      set_b,
   char const*     op_name)
{
   Tuple*      tuple_a;
   Tuple*      tuple_b;
   int         i;

   /* If one of the two involved sets is empty, the dimension of the
    * other one does not matter.
    */
   if (set_get_members(set_a) == 0 || set_get_members(set_b) == 0)
      return;
   
   if (set_get_dim(set_a) != set_get_dim(set_b))
   {
      fprintf(stderr, "*** Error 119: %s of sets with different dimension\n", op_name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   tuple_a = set_get_tuple(set_a, 0);
   tuple_b = set_get_tuple(set_b, 0);

   assert(tuple_get_dim(tuple_a) == set_get_dim(set_b));
   assert(tuple_get_dim(tuple_a) == tuple_get_dim(tuple_b));

   for(i = 0; i < tuple_get_dim(tuple_a); i++)
   {
      ElemType elem_type_a = elem_get_type(tuple_get_elem(tuple_a, i));
      ElemType elem_type_b = elem_get_type(tuple_get_elem(tuple_b, i));

      assert(elem_type_a == ELEM_NUMB || elem_type_a == ELEM_STRG);
      assert(elem_type_b == ELEM_NUMB || elem_type_b == ELEM_STRG);
      
      if (elem_type_a != elem_type_b)
      {
         fprintf(stderr, "*** Error 120: %s of sets with different types\n", op_name);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   tuple_free(tuple_a);
   tuple_free(tuple_b);
}

CodeNode* i_set_union(CodeNode* self)
{
   Set const*  set_a;
   Set const*  set_b;
   
   Trace("i_set_union");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   check_sets_compatible(self, set_a, set_b, "Union");
   
   code_value_set(self, set_union(set_a, set_b));

   return self;
}

CodeNode* i_set_union2(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   CodeNode*     lexpr;
   SetIter*      iter;
   Tuple*        tuple;
   Set*          set_r = NULL;
   Set*          set_old;
   Set const*    set_new;
   
   Trace("i_set_union2");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);
   
   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         if (set_r == NULL)
            set_r = set_copy(code_eval_child_set(self, 1));
         else
         {
            assert(set_r != NULL);
            
            set_old = set_copy(set_r);
            set_new = code_eval_child_set(self, 1);

            check_sets_compatible(self, set_old, set_new, "Union");

            set_free(set_r);

            set_r = set_union(set_old, set_new);

            set_free(set_old);
         }
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);

   if (set_r == NULL)
      set_r = set_empty_new(tuple_get_dim(pattern));
   
   code_value_set(self, set_r);

   return self;
}

CodeNode* i_set_minus(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;
   
   Trace("i_set_minus");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   check_sets_compatible(self, set_a, set_b, "Minus");
   
   code_value_set(self, set_minus(set_a, set_b));

   return self;
}

CodeNode* i_set_inter(CodeNode* self)
{
   Trace("i_set_inter");

   assert(code_is_valid(self));

   Set const* set_a = code_eval_child_set(self, 0);
   Set const* set_b = code_eval_child_set(self, 1);

   check_sets_compatible(self, set_a, set_b, "Intersection");

   code_value_set(self, set_inter(set_a, set_b));

   return self;
}

CodeNode* i_set_inter2(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   CodeNode*     lexpr;
   SetIter*      iter;
   Tuple*        tuple;
   Set*          set_r = NULL;
   Set*          set_old;
   Set const*    set_new;
   
   Trace("i_set_inter2");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);
   
   /* This routine is not efficient.
    * It would be better to make pairs and then unite the pairs, etc.
    * Now it is O(n) and it could be O(log n)
    */
   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         if (set_r == NULL)
            set_r = set_copy(code_eval_child_set(self, 1));
         else
         {
            set_old = set_copy(set_r);
            set_new = code_eval_child_set(self, 1);

            check_sets_compatible(self, set_old, set_new, "Intersection");

            set_free(set_r);

            set_r = set_inter(set_old, set_new);

            set_free(set_old);
         }
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);
   
   if (set_r == NULL)
      set_r = set_empty_new(tuple_get_dim(pattern));
   
   code_value_set(self, set_r);

   return self;
}


CodeNode* i_set_sdiff(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;
   
   Trace("i_set_sdiff");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   check_sets_compatible(self, set_a, set_b, "Symmetric Difference");

   code_value_set(self, set_sdiff(set_a, set_b));

   return self;
}

CodeNode* i_set_cross(CodeNode* self)
{
   Set const* set_a;
   Set const* set_b;
   
   Trace("i_set_cross");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   set_b = code_eval_child_set(self, 1);

   code_value_set(self, set_prod_new(set_a, set_b));

   return self;
}

CodeNode* i_set_range(CodeNode* self)
{
   int from;
   int upto;
   int step;
   int diff;
   
   Trace("i_set_range");

   assert(code_is_valid(self));

   from = checked_eval_numb_toint(self, 0, "123: \"from\" value");
   upto = checked_eval_numb_toint(self, 1, "124: \"upto\" value");
   step = checked_eval_numb_toint(self, 2, "125: \"step\" value");   
   diff = upto - from;
   step = Sgn(diff) * abs(step);

   if (diff == 0)
      step = 1;
   
   if (step == 0) 
   {
      fprintf(stderr, "*** Error 126: Zero \"step\" value in range\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   code_value_set(self, set_range_new(from, upto, step));

   return self;
}

CodeNode* i_set_range2(CodeNode* self)
{
   Trace("i_set_range2");

   assert(code_is_valid(self));

   int from = checked_eval_numb_toint(self, 0, "123: \"from\" value");
   int upto = checked_eval_numb_toint(self, 1, "124: \"upto\" value");
   int step = checked_eval_numb_toint(self, 2, "125: \"step\" value");   
   int diff = upto - from;

   if (step == 0) 
   {
      fprintf(stderr, "*** Error 126: Zero \"step\" value in range\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (((step > 0) && (diff < 0))
    || ((step < 0) && (diff > 0)))
      code_value_set(self, set_empty_new(1));
   else
      code_value_set(self, set_range_new(from, upto, step));

   return self;
}

static Set* heap_to_set(CodeNode const* self, Heap* heap, int dim)
{
   assert(code_is_valid(self));
   assert(heap_is_valid(heap));
   assert(dim >= 0);
   
   Set* set;

   if (heap_is_empty(heap))
   {
      if (stmt_trigger_warning(206))
      {
         fprintf(stderr, "--- Warning 206: argmin/argmax over empty set\n");
         code_errmsg(code_get_child(self, 0));
      }
      set = set_empty_new(dim);
   }
   else
   {
      Entry* entry = heap_pop_entry(heap);
      List*  list  = list_new_tuple(entry_get_tuple(entry));
      entry_free(entry);

      while(!heap_is_empty(heap))
      {
         entry = heap_pop_entry(heap);
         list_add_tuple(list, entry_get_tuple(entry));
         entry_free(entry);
      }
      set = set_new_from_list(list, SET_CHECK_WARN);

      list_free(list);
   }
   return set;
}

static int argmin_entry_cmp_descending(HeapData a, HeapData b)
{
   assert(entry_is_valid(a.entry));
   assert(entry_is_valid(b.entry));
   assert(entry_get_type(a.entry) == SYM_NUMB);  
   assert(entry_get_type(b.entry) == SYM_NUMB);
   
   return numb_cmp(entry_get_numb(b.entry), entry_get_numb(a.entry));
}

static int argmax_entry_cmp_ascending(HeapData a, HeapData b)
{
   assert(entry_is_valid(a.entry));
   assert(entry_is_valid(b.entry));
   assert(entry_get_type(a.entry) == SYM_NUMB); 
   assert(entry_get_type(b.entry) == SYM_NUMB);
   
   return numb_cmp(entry_get_numb(a.entry), entry_get_numb(b.entry));
}

static CodeNode* do_set_argminmax(CodeNode* self, bool is_min)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Numb const*   value;
   Heap*         heap;  
   int           size;
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 1);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);
   size    = checked_eval_numb_toint(self, 0, "207: \"size\" value");

   if (size < 1)
   {
      fprintf(stderr, "*** Error 208: \"size\" value %d not >= 1\n", size);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   /* There is no need for more entries than this
    */
   if (size > set_get_members(set))
      size = set_get_members(set);

   assert(size > 0);
   
   heap = heap_new_entry(size,
      is_min ? argmin_entry_cmp_descending : argmax_entry_cmp_ascending);

   assert(heap != NULL);

   warn_if_pattern_has_no_name(code_get_child(self, 1), pattern);

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         value = code_eval_child_numb(self, 2);      

         if (heap_is_full(heap))
         {
            assert(!heap_is_empty(heap));

            if (is_min)
            {
               /* Is value smaller than currently biggest in heap ?
                * In this case drop biggest element in heap
                */
               if (numb_cmp(value, entry_get_numb(heap_top_entry(heap))) < 0)
                  entry_free(heap_pop_entry(heap));
            }
            else
            {
               /* Is value bigger than currently smallest in heap ?
                * In this case drop smallest element in heap
                */
               if (numb_cmp(value, entry_get_numb(heap_top_entry(heap))) > 0)
                  entry_free(heap_pop_entry(heap));
            }
         }
         if (!heap_is_full(heap))
            heap_push_entry(heap, entry_new_numb(tuple, value));
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);

   code_value_set(self, heap_to_set(self, heap, tuple_get_dim(pattern)));

   heap_free(heap);
   
   return self;
}

CodeNode* i_set_argmin(CodeNode* self)
{
   Trace("i_set_argmin");
   
   return do_set_argminmax(self, true);
}

CodeNode* i_set_argmax(CodeNode* self)
{
   Trace("i_set_argmax");
   
   return do_set_argminmax(self, false);
}

CodeNode* i_set_proj(CodeNode* self)
{
   Set const*   set_a;
   Tuple const* tuple;
   int          dim;
   int          i;
   
   Trace("i_set_proj");

   assert(code_is_valid(self));

   set_a = code_eval_child_set(self, 0);
   tuple = code_eval_child_tuple(self, 1);
   dim   = set_get_dim(set_a);
   
   for(i = 0; i < tuple_get_dim(tuple); i++)
   {
      Elem const* elem = tuple_get_elem(tuple, i);
      Numb const* numb;
      int         idx;

      /* Are there only number in the selection tuple ?
       */
      if (ELEM_NUMB != elem_get_type(elem))
      {
         fprintf(stderr, "*** Error 127: Illegal value type in tuple: ");
         tuple_print(stderr, tuple);
         fprintf(stderr, " only numbers are possible\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      numb = elem_get_numb(elem);
      
      if (!numb_is_int(numb))
      {
         fprintf(stderr, "*** Error 128: Index value ");
         numb_print(stderr, numb);
         fprintf(stderr, " in proj too big or not an integer\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      idx = numb_toint(numb);
      
      /* Are all the number between 1 and dim(set) ?
       */
      if (idx < 1 || idx > dim)
      {
         fprintf(stderr, "*** Error 129: Illegal index %d, ", idx);
         fprintf(stderr, " set has only dimension %d\n", dim);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   code_value_set(self, set_proj(set_a, tuple));

   return self;
}

CodeNode* i_set_indexset(CodeNode* self)
{
   Symbol const* sym;
   
   Trace("i_set_indexset");

   assert(code_is_valid(self));

   sym = code_eval_child_symbol(self, 0);

   assert(sym != NULL);

   code_value_set(self, set_copy(symbol_get_iset(sym)));

   return self;
}

static int noneval_get_dim(CodeNode const* code_cexpr_or_tuple)
{
   assert(code_is_valid(code_cexpr_or_tuple));

   int dim = 1;
   
   /* Is it a tuple or a cexpr ?
    */
   if (code_get_inst(code_cexpr_or_tuple) == (Inst)i_tuple_new)
   {
      for(CodeNode const* code_cexpr_list = code_get_child(code_cexpr_or_tuple, 0);
          code_get_inst(code_cexpr_list) == (Inst)i_elem_list_add;
          code_cexpr_list = code_get_child(code_cexpr_list, 0))
      {
         dim++;
      }
   }
   return dim;
}

CodeNode* i_set_expr(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    iset;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   CodeNode*     cexpr_or_tuple;
   Elem*         elem          = NULL;
   List*         list          = NULL;
   bool          is_tuple_list = false;
   
   Trace("i_set_expr");

   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   iset    = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(iset, pattern);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while((tuple = set_iter_next(iter, iset)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
      {
         cexpr_or_tuple = code_eval_child(self, 1);      

         switch(code_get_type(cexpr_or_tuple))
         {
         case CODE_TUPLE :
            assert(list == NULL || is_tuple_list);

            is_tuple_list = true;
            break;
         case CODE_NUMB :
            assert(!is_tuple_list);

            elem = elem_new_numb(code_get_numb(cexpr_or_tuple));
            break;
         case CODE_STRG :
            assert(!is_tuple_list);
            
            elem = elem_new_strg(code_get_strg(cexpr_or_tuple));
            break;
         case CODE_NAME :
            assert(!is_tuple_list);

            fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
               code_get_name(cexpr_or_tuple));
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         default :
            abort();
         }
         assert(is_tuple_list || elem != NULL);

         if (is_tuple_list)
         {
            if (list == NULL)
               list = list_new_tuple(code_get_tuple(cexpr_or_tuple));
            else
               list_add_tuple(list, code_get_tuple(cexpr_or_tuple));
         }
         else
         {
            assert(elem != NULL);

            if (list == NULL)
               list = list_new_elem(elem);
            else
               list_add_elem(list, elem);

            elem_free(elem);
         }
      }
      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, iset);
   
   if (list == NULL)
   {
      if (stmt_trigger_warning(202))
      {
         fprintf(stderr, "--- Warning 202: Indexing over empty set\n");
         code_errmsg(code_get_child(self, 0));
      }
      /* If it is an cexpr list the dimension is 1, if it is a
       * tuple list, it is the dimension of the tuple.
       * Because of <i + j> we are not able to determine the dimension
       * of the tuple just by tuple_get_dim(code_eval_child_tuple(self, 1)).
       */
      code_value_set(self, set_empty_new(noneval_get_dim(code_get_child(self, 1))));
   }
   else
   {
      code_value_set(self, set_new_from_list(list, SET_CHECK_WARN));

      list_free(list);
   }
   return self;
}

/* ----------------------------------------------------------------------------
 * Tupel Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_tuple_new(CodeNode* self)
{
   List const* list  = code_eval_child_list(self, 0);
   int         n     = list_get_elems(list);
   Tuple*      tuple = tuple_new(n);
   ListElem*   le    = NULL;
   int         i;

   Trace("i_tuple_new");
   
   assert(code_is_valid(self));
   
   for(i = 0; i < n; i++)
   {
      Elem const* elem = list_get_elem(list, &le);

      assert(elem != NULL);

      tuple_set_elem(tuple, i, elem_copy(elem));
   }
   code_value_tuple(self, tuple);

   return self;
}

CodeNode* i_tuple_empty(CodeNode* self)
{
   Trace("i_tuple_empty");
   
   assert(code_is_valid(self));

   code_value_tuple(self, tuple_new(0));

   return self;
}

/* ----------------------------------------------------------------------------
 * Symbol Funktionen
 * ----------------------------------------------------------------------------
 */
static Set* set_from_idxset(IdxSet const* idxset)
{
   Tuple const*  pattern;
   Tuple*        tuple;
   Set*          newset;
   SetIter*      iter;
   Set const*    set;
   CodeNode*     lexpr;
   List*         list  = NULL;
   
   assert(idxset != NULL);
   
   set     = idxset_get_set(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   pattern = idxset_get_tuple(idxset);

   /* Is this an pseudo(idx)set ?
    */
   if (set_get_dim(set) == 0)
   {
      assert(tuple_get_dim(pattern) == 0);
      assert(code_get_bool(code_eval(lexpr)));

      newset = set_pseudo_new();
   }
   else if (idxset_is_unrestricted(idxset))
   {
      assert(code_get_bool(code_eval(lexpr)));

      newset = set_copy(set);
   }
   else
   {
      assert(tuple_get_dim(pattern) > 0);

      iter = set_iter_init(set, pattern);
      
      while((tuple = set_iter_next(iter, set)) != NULL)
      {
         local_install_tuple(pattern, tuple);

         if (code_get_bool(code_eval(lexpr)))
         {
            if (list == NULL)
               list = list_new_tuple(tuple);
            else
               list_add_tuple(list, tuple);
         }
         local_drop_frame();

         tuple_free(tuple);
      }
      set_iter_exit(iter, set);

      if (list == NULL)
      {
         newset = set_empty_new(tuple_get_dim(pattern));
         /* ??? maybe we need an error here ? */
      }
      else
      {
         newset = set_new_from_list(list, SET_CHECK_WARN);

         list_free(list);
      }
   }
   return newset;
}

CodeNode* i_newsym_set1(CodeNode* self)
{
   char const*   name;
   IdxSet const* idxset;
   Set*          iset;
   Symbol*       sym;

   Tuple const*  pattern;
   Tuple*        tuple;
   SetIter*      iter;
   
   Trace("i_newsym_set1");

   name    = code_eval_child_name(self, 0);
   idxset  = code_eval_child_idxset(self, 1);
   iset    = set_from_idxset(idxset);
   sym     = symbol_new(name, SYM_SET, iset, set_get_members(iset), NULL);

   assert(code_is_valid(self));

   if (set_get_members(iset) == 0)
   {
      fprintf(stderr, "*** Error 197: Empty index set for set\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   pattern = idxset_get_tuple(idxset);
   iter    = set_iter_init(iset, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 1), pattern);
   
   while((tuple = set_iter_next(iter, iset)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      symbol_add_entry(sym,
         entry_new_set(tuple,
            code_eval_child_set(self, 2)));

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, iset);
   set_free(iset);

   code_value_void(self);

   return self;
}
   
CodeNode* i_newsym_set2(CodeNode* self)
{
   char const*   name;
   IdxSet const* idxset;
   Set*          iset;
   Symbol*       sym;
   List const*   list;
   ListElem*     lelem;
   int           count;
   int           i;
   
   Trace("i_newsym_set2");

   assert(code_is_valid(self));

   name   = code_eval_child_name(self, 0);
   idxset = code_eval_child_idxset(self, 1);
   iset   = set_from_idxset(idxset);
   list   = code_eval_child_list(self, 2);
   count  = list_get_elems(list);

   assert(list_is_entrylist(list));

   /* Empty set ?
    */
   if (set_get_members(iset) == 0)
   {
      fprintf(stderr, "*** Error 197: Empty index set for set\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   
   /* Pseudo set ?
    */
   if (set_get_dim(iset) == 0)
   {
      set_free(iset);

      iset = set_new_from_list(list, SET_CHECK_WARN);
   }
   sym = symbol_new(name, SYM_SET, iset, count, NULL);

   lelem = NULL;
   
   for(i = 0; i < count; i++)
   {
      Entry const* entry  = list_get_entry(list, &lelem);
      Tuple const* tuple  = entry_get_tuple(entry);

      check_tuple_set_compatible(self, tuple, iset);

      if (set_lookup(iset, tuple))
         symbol_add_entry(sym, entry_copy(entry));
      else
      {
         fprintf(stderr, "*** Error 131: Illegal element ");
         tuple_print(stderr, tuple);
         fprintf(stderr, " for symbol\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   code_value_void(self);

   set_free(iset);
   
   return self;
}

static void insert_param_list_by_index(
   CodeNode const* self,
   Symbol*         sym,
   Set const*      iset,
   Tuple const*    pattern,
   List const*     list)
{
   SetIter*      iter;
   ListElem*     le_idx = NULL;
   int           count  = 0;
   Tuple*        tuple;
   int           list_entries;
   Entry*        new_entry;

   list_entries = list_get_elems(list); 
   iter         = set_iter_init(iset, pattern);
  
   while((tuple = set_iter_next(iter, iset)) != NULL && count < list_entries)
   {
      /* bool is not needed, because iset has only true elems
       */
      Entry const* entry = list_get_entry(list, &le_idx);

      switch(entry_get_type(entry))
      {
      case SYM_NUMB:
         new_entry = entry_new_numb(tuple, entry_get_numb(entry));
         break;
      case SYM_STRG :
         new_entry = entry_new_strg(tuple, entry_get_strg(entry));
         break;
      default :
         abort();
      }
      if (count > 0 && symbol_get_type(sym) != entry_get_type(new_entry))
      {
         fprintf(stderr, "*** Error 173: Illegal type in element ");
         entry_print(stderr, new_entry);
         fprintf(stderr, " for symbol\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      symbol_add_entry(sym, new_entry);
      
      tuple_free(tuple);
      
      count++;
   }
   if (tuple != NULL)
      tuple_free(tuple);
      
   set_iter_exit(iter, iset);

   if (count < list_entries)
   {
      if (stmt_trigger_warning(205))
      {
         fprintf(stderr,
            "--- Warning 205: %d excess entries for symbol %s ignored\n",
            list_entries - count,
            symbol_get_name(sym));      
         code_errmsg(self);
      }
   }
}

static void insert_param_list_by_list(
   CodeNode const* self,
   Symbol*         sym,
   Set const*      iset,
   List const*     list)
{
   ListElem*     le_idx = NULL;
   int           list_entries;
   int           i;

   list_entries = list_get_elems(list);
   
   for(i = 0; i < list_entries; i++)
   {
      Entry const* entry  = list_get_entry(list, &le_idx);
      Tuple const* tuple  = entry_get_tuple(entry);

      check_tuple_set_compatible(self, tuple, iset);

      if (!set_lookup(iset, tuple))
      {
         fprintf(stderr, "*** Error 134: Illegal element ");
         tuple_print(stderr, tuple);
         fprintf(stderr, " for symbol\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      if (i > 0 && symbol_get_type(sym) != entry_get_type(entry))
      {
         fprintf(stderr, "*** Error 173: Illegal type in element ");
         entry_print(stderr, entry);
         fprintf(stderr, " for symbol\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      symbol_add_entry(sym, entry_copy(entry));
   }
}

/* initialisation per list
 */
CodeNode* i_newsym_para1(CodeNode* self)
{
   char const*   name;
   IdxSet const* idxset;
   Set*          iset;
   List const*   list;
   CodeNode*     child3;
   Entry const*  deflt = ENTRY_NULL;
   int           list_entries;

   ListElem*     le_idx;
   Entry const*  entry;
   Tuple const*  tuple;
   
   Trace("i_newsym_para1");

   assert(code_is_valid(self));

   name   = code_eval_child_name(self, 0);
   idxset = code_eval_child_idxset(self, 1);
   iset   = set_from_idxset(idxset);
   list   = code_eval_child_list(self, 2);
   child3 = code_eval_child(self, 3);

   if (code_get_type(child3) != CODE_VOID)
      deflt = code_get_entry(code_eval(child3));

   if (!list_is_entrylist(list))
   {
      /* This errors occurs, if the parameter is missing in the template
       * for a "read" statement.
       */
      assert(list_is_tuplelist(list));
      
      fprintf(stderr, "*** Error 132: Values in parameter list missing,\n");
      fprintf(stderr, "               probably wrong read template\n");      
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   
   /* First element will determine the type (see SYM_ERR below)
    */
   list_entries = list_get_elems(list);

   /* I found no way to make the following error happen.
    * You will get either an error 157 or an parse error.
    * In case there is a way a parse error with
    * message "Symbol xxx not initialised" will result.
    * In this case the code below should be reactivated.
    */
   assert(list_entries > 0);
#if 0 /* ??? */
   /* So if there is no first element, we are in trouble.
    */
   if (list_entries == 0)
   {
      fprintf(stderr, "*** Error xxx: Empty initialisation for parameter \"%s\n",
         name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
#endif
   le_idx = NULL;

   /* Now there is the question if we got an entry list with index tuples or without.
    * A special case is the singleton, i.e. an entry for a single parameter value.
    */
   entry  = list_get_entry(list, &le_idx);
   tuple  = entry_get_tuple(entry);

   /* Check whether the file was empty
    */
   if (entry_get_type(entry) == SYM_SET)
   {
      (void)symbol_new(name, deflt == ENTRY_NULL ? SYM_ERR : entry_get_type(deflt), iset, 0, deflt);
   }
   else
   {
      Symbol* sym;

      if (set_get_members(iset) == 0)
      {
         fprintf(stderr, "*** Error 135: Empty index set for parameter\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      sym = symbol_new(name, SYM_ERR, iset, list_entries, deflt);

      if (list_entries > 1 && tuple_get_dim(tuple) == 0 && set_get_dim(iset) > 0)
      {
         insert_param_list_by_index(self, sym, iset, idxset_get_tuple(idxset), list);
      }
      else
      {
         insert_param_list_by_list(self, sym, iset, list);
      }
   }
   code_value_void(self);

   set_free(iset);

   return self;
}

/* initialisation per element
 */
CodeNode* i_newsym_para2(CodeNode* self)
{
   char const*   name;
   Set*          iset;
   IdxSet const* idxset;
   Symbol*       sym;
   Entry*        entry;
   Tuple*        tuple;
   Tuple const*  pattern;
   SetIter*      iter;
   int           count = 0;
   
   Trace("i_newsym_para2");

   assert(code_is_valid(self));

   name    = code_eval_child_name(self, 0);
   idxset  = code_eval_child_idxset(self, 1);
   iset    = set_from_idxset(idxset);

   /* Since we alway initialise all elements, there is no use to evaluate the
    * default parameter.
    */

   if (set_get_members(iset) == 0)
   {
      fprintf(stderr, "*** Error 135: Index set for parameter \"%s\" is empty\n",
         name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   
   sym     = symbol_new(name, SYM_ERR, iset, set_get_members(iset), ENTRY_NULL);
   pattern = idxset_get_tuple(idxset);
   iter    = set_iter_init(iset, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 1), pattern);
   
   while((tuple = set_iter_next(iter, iset)) != NULL)
   {
      /* bool is not needed, because iset has only true elems
       */
      local_install_tuple(pattern, tuple);

      CodeNode* child = code_eval_child(self, 2);

      switch(code_get_type(child))
      {
      case CODE_NUMB:
         entry = entry_new_numb(tuple, code_get_numb(child));
         break;
      case CODE_STRG :
         entry = entry_new_strg(tuple, code_get_strg(child));
         break;
      case CODE_NAME :
         fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
            code_get_name(child));
         code_errmsg(code_get_child(self, 2));
         zpl_exit(EXIT_FAILURE);
      default :
         abort();
      }
      if (count > 0 && symbol_get_type(sym) != entry_get_type(entry))
      {
         fprintf(stderr, "*** Error 173: Illegal type in element ");
         entry_print(stderr, entry);
         fprintf(stderr, " for symbol\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      symbol_add_entry(sym, entry);
      
      local_drop_frame();

      tuple_free(tuple);
      
      count++;
   }
   set_iter_exit(iter, iset);

   code_value_void(self);

   set_free(iset);

   return self;
}

CodeNode* i_newsym_var(CodeNode* self)
{
   char const*   name;
   IdxSet const* idxset;
   Set*          iset;
   Symbol*       sym;
   Tuple*        tuple;
   Tuple const*  pattern;
   VarClass      varclass;
   SetIter*      iter;
   Numb*         temp;
   
   Trace("i_newsym_var");

   assert(code_is_valid(self));

   name     = code_eval_child_name(self, 0);
   idxset   = code_eval_child_idxset(self, 1);
   varclass = code_eval_child_varclass(self, 2);
   iset     = set_from_idxset(idxset);
   pattern  = idxset_get_tuple(idxset);
   sym      = symbol_new(name, SYM_VAR, iset, set_get_members(iset), NULL);
   iter     = set_iter_init(iset, pattern);

   warn_if_pattern_has_no_name(code_get_child(self, 1), pattern);
   
   while((tuple = set_iter_next(iter, iset)) != NULL)
   {      
      Bound* lower;
      Bound* upper;
      Numb const* priority;
      Numb const* startval;
      
      local_install_tuple(pattern, tuple);

      lower         = bound_copy(code_eval_child_bound(self, 3));
      upper         = bound_copy(code_eval_child_bound(self, 4));
      priority = code_eval_child_numb(self, 5);
      startval = code_eval_child_numb(self, 6);
      
      /* Parser makes sure, cannot happen
       */
      assert(bound_get_type(lower) != BOUND_INFTY);
      assert(bound_get_type(upper) != BOUND_MINUS_INFTY);

      /* Integral bounds for integral variables ?
       */
      if (varclass != VAR_CON)
      {
         if (bound_get_type(lower) == BOUND_VALUE)
         {
            temp = numb_copy(bound_get_value(lower));
            numb_ceil(temp);
      
            if (!numb_equal(temp, bound_get_value(lower)))
            {
               bound_free(lower);
               lower = bound_new(BOUND_VALUE, temp);
               
               if (stmt_trigger_warning(139))
               {
                  fprintf(stderr,
                     "--- Warning 139: Lower bound for integral var %s truncated to ",
                     name);
                  numb_print(stderr, temp);
                  fputc('\n', stderr);
               }
            }
            numb_free(temp);
         }
         if (bound_get_type(upper) == BOUND_VALUE)
         {
            temp = numb_copy(bound_get_value(upper));
            numb_floor(temp);
      
            if (!numb_equal(temp, bound_get_value(upper)))
            {
               bound_free(upper);
               upper = bound_new(BOUND_VALUE, temp);
               
               if (stmt_trigger_warning(140))
               {
                  fprintf(stderr,
                     "--- Warning 140: Upper bound for integral var %s truncated to ",
                     name);
                  numb_print(stderr, temp);
                  fputc('\n', stderr);
               }
            }
            numb_free(temp);
         }
      }
      if (  bound_get_type(lower) == BOUND_VALUE
         && bound_get_type(upper) == BOUND_VALUE
         && numb_cmp(bound_get_value(lower), bound_get_value(upper)) > 0)
      {
         fprintf(stderr, "*** Error 141: Infeasible due to conflicting bounds for var %s\n",
            name);
         fprintf(stderr, "               lower=%g > upper=%g\n",
            numb_todbl(bound_get_value(lower)),
            numb_todbl(bound_get_value(upper)));
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }

      /* Hier geben wir der Variable einen eindeutigen Namen
       */
      char* tuplestr = tuple_tostr(tuple);
      char* varname  = malloc(strlen(name) + strlen(tuplestr) + 2);

      assert(varname != NULL);
      
      sprintf(varname, "%s%s", name, tuplestr);

      /* Und nun legen wir sie an.
       */
      Var* var = xlp_addvar(prog_get_lp(), varname, varclass, lower, upper, priority, startval);

      symbol_add_entry(sym, entry_new_var(tuple, var));

      free(varname);
      free(tuplestr);
      
      local_drop_frame();

      tuple_free(tuple);
      bound_free(lower);
      bound_free(upper);
   }
   set_iter_exit(iter, iset);

   code_value_void(self);

   set_free(iset);

   return self;
}

CodeNode* i_symbol_deref(CodeNode* self)
{
   Symbol const* sym;
   Tuple const*  tuple;
   Entry const*  entry;
   Term*         term;
   int           i;
   
   Trace("i_symbol_deref");
   
   assert(code_is_valid(self));

   sym   = code_eval_child_symbol(self, 0);
   tuple = code_eval_child_tuple(self, 1);   

   /* wurde schon in mmlscan ueberprueft
    */
   assert(sym != NULL);

   for(i = 0; i < tuple_get_dim(tuple); i++)
   {
      Elem const* elem = tuple_get_elem(tuple, i);

      /* Are there any unresolved names in the tuple?
       */
      if (ELEM_NAME == elem_get_type(elem))
      {
         fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
            elem_get_name(elem));
         code_errmsg(code_get_child(self, 1));
         zpl_exit(EXIT_FAILURE);
      }
   }

   /* Tuple might not fit to symbol
    */
   entry = symbol_lookup_entry(sym, tuple);

   if (NULL == entry)
   {
      fprintf(stderr, "*** Error 142: Unknown index ");
      tuple_print(stderr, tuple);
      fprintf(stderr, " for symbol \"%s\"\n", symbol_get_name(sym));
      code_errmsg(code_get_child(self, 1));
      zpl_exit(EXIT_FAILURE);
   }
   
   switch(symbol_get_type(sym))
   {
   case SYM_NUMB :
      code_value_numb(self, numb_copy(entry_get_numb(entry)));
      break;
   case SYM_STRG :
      code_value_strg(self, entry_get_strg(entry));
      break;
   case SYM_SET :
      code_value_set(self, set_copy(entry_get_set(entry)));
      break;
   case SYM_VAR :
      term = term_new(1);
      term_add_elem(term, entry, numb_one(), MFUN_NONE);
      code_value_term(self, term);
      break;
   default :
      abort();
   }
   return self;
}

CodeNode* i_term_mul(CodeNode* self)
{
   Term const* term_a;
   Term const* term_b;

   assert(code_is_valid(self));

   term_a = code_eval_child_term(self, 0);
   term_b = code_eval_child_term(self, 1);

   code_value_term(self, term_mul_term(term_a, term_b));

   return self;
}

CodeNode* i_term_power(CodeNode* self)
{
   int           power;
   Term const*   term;
   Term*         term_result;
   Term*         term_temp;
   
   Trace("i_term_power");
   
   assert(code_is_valid(self));

   term  = code_eval_child_term(self, 0);
   power = checked_eval_numb_toint(self, 1, "112: Exponent value");

   if (power < 0)
   {
      fprintf(stderr, "*** Error 121: Negative exponent on variable\n");
      code_errmsg(code_get_child(self, 0));
      zpl_exit(EXIT_FAILURE);
   }
   
   if (power == 0)
   {
      term_result = term_new(1);
      term_add_constant(term_result, numb_one());
   } 
   else 
   {
      term_result = term_copy(term);

      for(int i = 1; i < power; i++)
      {   
         term_temp = term_mul_term(term_result, term);
         term_free(term_result);
         term_result = term_temp;
      }
   }
   code_value_term(self, term_result);

   return self;
}

CodeNode* i_newdef(CodeNode* self)
{
   Define* def;
   
   Trace("i_newdef");

   assert(code_is_valid(self));

   def = code_eval_child_define(self, 0);

   define_set_param(def, tuple_copy(code_eval_child_tuple(self, 1)));
   define_set_code(def, code_get_child(self, 2));
   
   code_value_void(self);

   return self;
}

CodeNode* i_define_deref(CodeNode* self)
{
   Define const* def;
   Tuple const*  tuple;
   Tuple const*  param;
   int           i;
   
   Trace("i_define_deref");
   
   assert(code_is_valid(self));

   def   = code_eval_child_define(self, 0);
   tuple = code_eval_child_tuple(self, 1);   

   for(i = 0; i < tuple_get_dim(tuple); i++)
   {
      ElemType elem_type = elem_get_type(tuple_get_elem(tuple, i));
      
      if (elem_type != ELEM_NUMB && elem_type != ELEM_STRG)
      {
         assert(elem_type == ELEM_NAME);
         
         fprintf(stderr, "*** Error 170: Uninitialised local parameter \"%s\"\n",
            elem_get_name(tuple_get_elem(tuple, i)));
         fprintf(stderr, "               in call of define \"%s\".\n",
            define_get_name(def));
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   
   /* wurde schon in mmlscan ueberprueft
    */
   assert(def != NULL);

   param = define_get_param(def);
   
   if (tuple_get_dim(tuple) != tuple_get_dim(param))
   {
      fprintf(stderr, "*** Error 171: Wrong number of arguments (%d instead of %d)\n",
         tuple_get_dim(tuple),
         tuple_get_dim(param));
      fprintf(stderr, "               for call of define \"%s\".\n",
         define_get_name(def));
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   local_install_tuple(param, tuple);

   code_copy_value(self, code_eval(define_get_code(def)));

   local_drop_frame();

   return self;
}


/* ----------------------------------------------------------------------------
 * Index Set Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_set_idxset(CodeNode* self)
{
   IdxSet const* idxset;

   Trace("i_set_idxset");

   idxset = code_eval_child_idxset(self, 0);
   
   code_value_set(self, set_from_idxset(idxset));

   return self;
}

CodeNode* i_idxset_new(CodeNode* self)
{
   assert(code_is_valid(self));

   Tuple*       tuple;
   Tuple*       t0;
   CodeNode*    lexpr;
   Set const*   set;
   int          dim;
   bool         is_unrestricted;
         
   Trace("i_idxset_new");


   t0     = tuple_new(0);
   tuple  = tuple_copy(code_eval_child_tuple(self, 0));
   set    = code_eval_child_set(self, 1);
   lexpr  = code_get_child(self, 2);
   dim    = set_get_dim(set);

   is_unrestricted = code_get_inst(lexpr) == (Inst)i_bool_true;

   /* If we get any empty set with dimension 0, it is not the result
    * of some other operation, but genuine empty.
    * This is gives a warning.
    */
   if (dim == 0)
   {
      assert(set_get_members(set) == 0);

      if (stmt_trigger_warning(195))
      {
         fprintf(stderr, "--- Warning 195: Genuine empty index as index set\n");
         code_errmsg(self);
      }
   }
   /* Attention: set_get_members(set) == 0 is possible!
    */
   
   /* If no index tuple was given, we construct one.
    * This will always be ok.
    */
   if (!tuple_cmp(tuple, t0))
   {
      tuple_free(tuple);

      tuple = tuple_new(dim);
      
      for(int i = 0; i < dim; i++)
      {
         char name[13]; /* "@-2000000000" */

         sprintf(name, "@%d", i + 1);
         tuple_set_elem(tuple, i, elem_new_name(str_new(name)));
      }
   }
   else
   {
      /* If a index tuple was given, check if
       * - the dimension is correct, sets of dim 0 fit any tuple
       * - any not NAME type entries are compatible.
       * - the set is unrestricted, ie all NAMES, no WITH.
       */    
      if (dim > 0 && tuple_get_dim(tuple) != dim)
      {
         fprintf(stderr, "*** Error 188: Index tuple ");
         tuple_print(stderr, tuple);
         fprintf(stderr, " has wrong dimension %d, expected %d\n",
            tuple_get_dim(tuple),
            dim);

         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      if (set_get_members(set) > 0)
      {
         Tuple* t1 = set_get_tuple(set, 0);

         for(int i = 0; i < dim; i++)
         {
            ElemType elem_type = elem_get_type(tuple_get_elem(tuple, i));
               
            if (elem_type != ELEM_NAME)
            {
               is_unrestricted = false;
               
               if (elem_type != elem_get_type(tuple_get_elem(t1, i)))
               {
                  fprintf(stderr, "*** Error 198: Incompatible index tuple\n");
                  tuple_print(stderr, tuple);
                  fprintf(stderr, " component %d is not compatible with ", i + 1);
                  tuple_print(stderr, t1);
                  fprintf(stderr, "\n");
                  code_errmsg(self);
                  zpl_exit(EXIT_FAILURE);
               }
            }
         }
         tuple_free(t1);
      }
   }
   code_value_idxset(self, idxset_new(tuple, set, lexpr, is_unrestricted));

   tuple_free(t0);
   tuple_free(tuple);

   return self;
}

CodeNode* i_idxset_pseudo_new(CodeNode* self)
{
   Tuple*    tuple;
   Set*      set;
         
   Trace("i_idxset_pseudo_new");

   assert(code_is_valid(self));

   tuple = tuple_new(0);
   set   = set_pseudo_new();
   
   code_value_idxset(self, idxset_new(tuple, set, code_get_child(self, 0), true));

   set_free(set);
   tuple_free(tuple);

   return self;
}

/* ----------------------------------------------------------------------------
 * Local Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_local_deref(CodeNode* self)
{
   char const* name;
   Elem const* elem;
   
   Trace("i_local_deref");
   
   assert(code_is_valid(self));

   name = code_eval_child_name(self, 0);   
   elem = local_lookup(name);

   if (elem == NULL)
      code_value_name(self, name);
   else
   {
      switch(elem_get_type(elem))
      {
      case ELEM_NUMB :
         code_value_numb(self, numb_copy(elem_get_numb(elem)));
         break;
      case ELEM_STRG :
         code_value_strg(self, elem_get_strg(elem));
         break;
      default :
         abort();
      }
   }
   return self;
}   

/* ----------------------------------------------------------------------------
 * Term Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_term_coeff(CodeNode* self)
{
   Term*       term;
   Numb const* coeff;
   
   Trace("i_term_coeff");
   
   assert(code_is_valid(self));

   term  = term_copy(code_eval_child_term(self, 0));
   coeff = code_eval_child_numb(self, 1);

   term_mul_coeff(term, coeff);
   
   code_value_term(self, term);

   return self;
}

CodeNode* i_term_const(CodeNode* self)
{
   Term*       term;
   Numb const* numb;
   
   Trace("i_term_const");
   
   assert(code_is_valid(self));

   term = term_copy(code_eval_child_term(self, 0));
   numb = code_eval_child_numb(self, 1);

   term_add_constant(term, numb);
   
   code_value_term(self, term);

   return self;
}

CodeNode* i_term_add(CodeNode* self)
{
   Term const* term_b;
   CodeNode*   child0;

   Trace("i_term_add");
   
   assert(code_is_valid(self));

   term_b = code_eval_child_term(self, 1);
   child0 = code_get_child(self, 0);

   if (term_get_elements(term_b) == 1)
   {
      (void)code_eval(child0);

      term_append_term(code_value_steal_term(self, 0), term_b);
   }
   else
   {
      code_value_term(self, term_add_term(code_eval_child_term(self, 0), term_b));
   
      code_free_value(child0);
   }
   code_free_value(code_get_child(self, 1));

   return self;
}

CodeNode* i_term_sub(CodeNode* self)
{
   Term const* term_a;
   Term const* term_b;
   
   Trace("i_term_sub");
   
   assert(code_is_valid(self));

   term_a = code_eval_child_term(self, 0);
   term_b = code_eval_child_term(self, 1);
   
   code_value_term(self, term_sub_term(term_a, term_b));

   return self;
}

CodeNode* i_term_sum(CodeNode* self)
{
   IdxSet const* idxset;
   Set const*    set;
   Tuple const*  pattern;
   Tuple*        tuple;
   CodeNode*     lexpr;
   SetIter*      iter;
   Term*         term_r;

   Trace("i_term_sum");
   
   assert(code_is_valid(self));

   idxset  = code_eval_child_idxset(self, 0);
   set     = idxset_get_set(idxset);
   pattern = idxset_get_tuple(idxset);
   lexpr   = idxset_get_lexpr(idxset);
   iter    = set_iter_init(set, pattern);
   term_r  = term_new(1);
   
   warn_if_pattern_has_no_name(code_get_child(self, 0), pattern);

   while((tuple = set_iter_next(iter, set)) != NULL)
   {
      local_install_tuple(pattern, tuple);

      if (code_get_bool(code_eval(lexpr)))
         term_append_term(term_r, code_eval_child_term(self, 1));

      local_drop_frame();

      tuple_free(tuple);
   }
   set_iter_exit(iter, set);
   
   code_value_term(self, term_r);

   return self;
}

CodeNode* i_term_expr(CodeNode* self)
{
   Term*       term;
   Numb const* numb;
   
   Trace("i_term_expr");
   
   assert(code_is_valid(self));

   term  = term_new(1);
   numb  = code_eval_child_numb(self, 0);

   term_add_constant(term, numb);
   
   code_value_term(self, term);

   return self;
}

/* ----------------------------------------------------------------------------
 * Entry Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_entry(CodeNode* self)
{
   Tuple const* tuple;
   Entry*       entry;
   CodeNode*    child;
   
   Trace("i_entry");
   
   assert(code_is_valid(self));

   tuple = code_eval_child_tuple(self, 0);
   child = code_eval_child(self, 1);

   switch(code_get_type(child))
   {
   case CODE_NUMB :
      entry = entry_new_numb(tuple, code_get_numb(child));
      break;
   case CODE_STRG :
      entry = entry_new_strg(tuple, code_get_strg(child));
      break;
   case CODE_SET :
      entry = entry_new_set(tuple, code_get_set(child));
      break;
   case CODE_NAME :
      fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
         code_get_name(child));
      code_errmsg(child);
      zpl_exit(EXIT_FAILURE);
   default :
      abort();
   }
   code_value_entry(self, entry);

   return self;
}

/* ----------------------------------------------------------------------------
 * List Funktionen
 * ----------------------------------------------------------------------------
 */
CodeNode* i_elem_list_new(CodeNode* self)
{
   CodeNode* child = code_eval_child(self, 0);
   Elem*     elem;
   
   Trace("i_elem_list_new");

   assert(code_is_valid(self));

   switch(code_get_type(child))
   {
   case CODE_NUMB :
      elem = elem_new_numb(code_get_numb(child));
      break;
   case CODE_STRG :
      elem = elem_new_strg(code_get_strg(child));
      break;
   case CODE_NAME :
      elem = elem_new_name(code_get_name(child));
      break;
   default :
      abort();
   }
   code_value_list(self, list_new_elem(elem));

   elem_free(elem);

   return self;
}

static Elem* make_elem(CodeNode* node)
{
   Elem* elem;

   assert(code_is_valid(node));

   switch(code_get_type(node))
   {
   case CODE_NUMB :
      elem = elem_new_numb(code_get_numb(node));
      break;
   case CODE_STRG :
      elem = elem_new_strg(code_get_strg(node));
      break;
   case CODE_NAME :
      elem = elem_new_name(code_get_name(node));
      break;
   default :
      abort();
   }
   return elem;
}

CodeNode* i_elem_list_add(CodeNode* self)
{
   CodeNode* node;
   Elem*     elem;
   List*     list;
      
   Trace("i_elem_list_add2");

   assert(code_is_valid(self));

   elem = make_elem(code_eval_child(self, 1));   
   list = list_new_elem(elem);
   elem_free(elem);
   
   node  = code_get_child(self, 0);

   /* If true, process directly without using a lot of recursion
    */
   while(code_get_inst(node) == (Inst)i_elem_list_add)
   {      
      elem = make_elem(code_eval_child(node, 1));
      list_insert_elem(list, elem);
      elem_free(elem);
      
      node = code_get_child(node, 0);
   }
   /* This is likely, so make it the fast way
    */
   if (code_get_inst(node) == (Inst)i_elem_list_new)
   {
      elem = make_elem(code_eval_child(node, 0));
      list_insert_elem(list, elem);
      elem_free(elem);
   }
   else
   {
      ListElem*   l = NULL;
      List const* head_list;
      Elem const* celem;
      
      head_list = code_get_list(code_eval(node));
      
      while(NULL != (celem = list_get_elem(head_list, &l)))
         list_insert_elem(list, celem);
   }
   code_value_list(self, list);

   return self;
}

CodeNode* i_tuple_list_new(CodeNode* self)
{
   Trace("i_tuple_list_new");

   assert(code_is_valid(self));

   code_value_list(self,
      list_new_tuple(
         code_eval_child_tuple(self, 0)));

   return self;
}

CodeNode* i_tuple_list_add(CodeNode* self)
{
   CodeNode*    node;
   Tuple const* tuple;
   List*        list;
   
   Trace("i_tuple_list_add2");

   assert(code_is_valid(self));

   node  = code_get_child(self, 0);
   tuple = code_eval_child_tuple(self, 1);
   list  = list_new_tuple(tuple);

   /* If true, process directly without using a lot of recursion
    */
   while(code_get_inst(node) == (Inst)i_tuple_list_add)
   {      
      list_insert_tuple(list, code_eval_child_tuple(node, 1));
      
      node = code_get_child(node, 0);
   }
   /* This is likely, so make it the fast way
    */
   if (code_get_inst(node) == (Inst)i_tuple_list_new)
      list_insert_tuple(list, code_eval_child_tuple(node, 0));
   else
   {
      ListElem*   l = NULL;
      List const* head_list;
      
      head_list = code_get_list(code_eval(node));
      
      while(NULL != (tuple = list_get_tuple(head_list, &l)))
         list_insert_tuple(list, tuple);
   }
   code_value_list(self, list);

   return self;
}

CodeNode* i_entry_list_new(CodeNode* self)
{
   Trace("i_entry_list_new");

   assert(code_is_valid(self));

   code_value_list(self,
      list_new_entry(
         code_eval_child_entry(self, 0)));

   return self;
}

CodeNode* i_entry_list_add(CodeNode* self)
{
   Entry const* entry;
   List*        list;
   CodeNode*    node;
   
   Trace("i_entry_list_add3");

   assert(code_is_valid(self));

   node  = code_get_child(self, 0);
   entry = code_eval_child_entry(self, 1);
   list  = list_new_entry(entry);

   /* If true, process directly without using a lot of recursion
    */
   while(code_get_inst(node) == (Inst)i_entry_list_add)
   {      
      list_insert_entry(list, code_eval_child_entry(node, 1));
      
      node = code_get_child(node, 0);
   }
   /* This is likely, so make it the fast way
    */
   if (code_get_inst(node) == (Inst)i_entry_list_new)
      list_insert_entry(list, code_eval_child_entry(node, 0));
   else
   {
      ListElem*   l = NULL;
      List const* head_list;
      
      head_list = code_get_list(code_eval(node));
      
      while(NULL != (entry = list_get_entry(head_list, &l)))
         list_insert_entry(list, entry);
   }
   code_value_list(self, list);
   
   return self;
}

CodeNode* i_entry_list_subsets(CodeNode* self)
{
   Set const* set;
   List*      list = NULL;
   int        subset_size_from;
   int        subset_size_to;
   SetIterIdx idx  = 0;
   int        used;
   int        i;
   
   Trace("i_entry_list_subsets");

   assert(code_is_valid(self));

   set              = code_eval_child_set(self, 0);
   used             = set_get_members(set);
   subset_size_from = checked_eval_numb_toint(self, 1, "143: Size for subsets");
   subset_size_to   = checked_eval_numb_toint(self, 2, "143: Size for subsets");

   if (subset_size_to == -1)
      subset_size_to = subset_size_from < used ? subset_size_from : used;

   if (used < 1)
   {
      fprintf(stderr, "*** Error 144: Tried to build subsets of empty set\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(set_get_dim(set) > 0);
   
   if (subset_size_from < 1 || subset_size_from > subset_size_to)
   {
      fprintf(stderr, "*** Error 145: Illegal size for subsets %d,\n", subset_size_from);
      fprintf(stderr, "               should be between 1 and %d\n", subset_size_to);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   if (subset_size_to > used)
   {
      fprintf(stderr, "*** Error 220: Illegal size for subsets %d,\n", subset_size_to);
      fprintf(stderr, "               should be between %d and %d\n", subset_size_from, used);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   for(i = subset_size_from; i <= subset_size_to; i++)
      list = set_subsets_list(set, i, list, &idx);

   assert(list != NULL);

   code_value_list(self, list);

   return self;
}

CodeNode* i_entry_list_powerset(CodeNode* self)
{
   Set const* set;
   List*      list = NULL;
   SetIterIdx idx  = 0;
   int        i;
   int        used;

   Trace("i_entry_list_powerset");

   assert(code_is_valid(self));

   set  = code_eval_child_set(self, 0);
   used = set_get_members(set);
   
   if (used < 1)
   {
      fprintf(stderr, "*** Error 146: Tried to build powerset of empty set\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   assert(set_get_dim(set) > 0);

   for(i = 0; i <= used; i++)
      list = set_subsets_list(set, i, list, &idx);

   assert(list != NULL);
   
   code_value_list(self, list);
   
   return self;
}

CodeNode* i_list_matrix(CodeNode* self)
{
   Trace("i_list_matrix");

   assert(code_is_valid(self));

   List const* head_list  = code_eval_child_list(self, 0);
   List const* body_list  = code_eval_child_list(self, 1);
   List*       list       = NULL;
   ListElem*   le_body    = NULL;
   int         head_count = list_get_elems(head_list);
   int         body_count = list_get_elems(body_list);
   
   assert(head_count > 0);
   assert(body_count > 0);
   assert(body_count % 2 == 0); /* has to be even */

   for(int i = 0; i < body_count; i += 2)
   {
      List const* idx_list  = list_get_list(body_list, &le_body);
      List const* val_list  = list_get_list(body_list, &le_body);
      int         idx_count = list_get_elems(idx_list);
      ListElem*   le_head   = NULL;
      ListElem*   le_val    = NULL;

      /* Number of values in a lines has to be equal the
       * number of elements in the head list
       */
      if (list_get_elems(val_list) != head_count)
      {
         fprintf(stderr, "*** Error 172: Wrong number of entries (%d) in table line,\n",
            list_get_elems(val_list));
         fprintf(stderr, "               expected %d entries\n", head_count);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }

      /* For each element in the head list we end up with
       * one element in the result list
       */
      for(int j = 0; j < head_count; j++)
      {
         /* Construct tuple. If idx_count is not constant, we will later
          * get an error when the list is applied to the parameter
          */
         Tuple* tuple = tuple_new(idx_count + 1);
         Entry* entry;
         int    k;
         
         ListElem* le_idx = NULL;
         for(k = 0; k < idx_count; k++)
            tuple_set_elem(tuple, k, elem_copy(list_get_elem(idx_list, &le_idx)));

         tuple_set_elem(tuple, k, elem_copy(list_get_elem(head_list, &le_head)));
      
         Elem const* const elem = list_get_elem(val_list, &le_val);

         assert(elem != NULL);
         
         switch(elem_get_type(elem))
         {
         case ELEM_NUMB :
            entry = entry_new_numb(tuple, elem_get_numb(elem));
            break;
         case ELEM_STRG :
            entry = entry_new_strg(tuple, elem_get_strg(elem));
            break;
         case ELEM_NAME :
            fprintf(stderr, "*** Error 133: Unknown symbol \"%s\"\n",
               elem_get_name(elem));
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         default :
            abort();
         }
         if (list == NULL)
            list  = list_new_entry(entry);
         else
            list_add_entry(list, entry);

         entry_free(entry);
         tuple_free(tuple);
      }
   }
   assert(list != NULL);
   
   code_value_list(self, list);

   return self;
}

CodeNode* i_matrix_list_new(CodeNode* self)
{
   List* list;
   
   Trace("i_matrix_list_new");

   assert(code_is_valid(self));

   list = list_new_list(code_eval_child_list(self, 0));
   list_add_list(list, code_eval_child_list(self, 1));

   code_value_list(self, list);

   return self;
}

CodeNode* i_matrix_list_add(CodeNode* self)
{
   List* list;
  
   Trace("i_matrix_list_add");

   assert(code_is_valid(self));

   list = list_copy(code_eval_child_list(self, 0));
   list_add_list(list, code_eval_child_list(self, 1));
   list_add_list(list, code_eval_child_list(self, 2));

   code_value_list(self, list);

   return self;
}

static void objective(CodeNode* self, bool minimize)
{
   char const* name;
   Term const* term;
   
   assert(code_is_valid(self));

   name = code_eval_child_name(self, 0);

   if (!conname_set(name))
   {
      fprintf(stderr, "*** Error 105: Duplicate constraint name \"%s\"\n", name);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   term = code_eval_child_term(self, 1);

   if (xlp_setobj(prog_get_lp(), name, minimize))
   {
      fprintf(stderr, "--- Warning 223: Objective function %s overwrites existing one\n", name);
      code_errmsg(self);
   }
    xlp_addtoobj(prog_get_lp(), term);

   conname_free();

   code_value_void(self);
}

CodeNode* i_object_min(CodeNode* self)
{
   Trace("i_object_min");

   assert(code_is_valid(self));

   objective(self, true);

   return self;
}

CodeNode* i_object_max(CodeNode* self)
{
   Trace("i_object_max");

   assert(code_is_valid(self));

   objective(self, false);

   return self;
}

CodeNode* i_print(CodeNode* self)
{
   CodeNode* child;
   
   Trace("i_print");

   assert(code_is_valid(self));

   child = code_eval(code_get_child(self, 0));

   switch(code_get_type(child))
   {
   case CODE_LIST :
      {
         List const* list = code_get_list(child);
         ListElem*   le = NULL;
         Elem const* elem;

         assert(list_is_elemlist(list));
         
         while(NULL != (elem = list_get_elem(list, &le)))
            elem_print(stdout, elem, false);
      }
      break;      
   case CODE_TUPLE :
      tuple_print(stdout, code_get_tuple(child));
      break;
   case CODE_SET :
      set_print(stdout, code_get_set(child));
      break;
   case CODE_SYM :
      symbol_print(stdout, code_get_symbol(child));
      break;
   case CODE_BOOL :
      fputs(code_get_bool(child) ? "true" : "false", stdout); 
      break;
   default :
      abort();
   }
   fputc('\n', stdout);
   
   code_value_void(self);

   return self;
}

CodeNode* i_bound_new(CodeNode* self)
{
   Trace("i_bound_new");

   assert(code_is_valid(self));

   code_value_bound(self,
      bound_new(BOUND_VALUE,
         code_eval_child_numb(self, 0)));

   return self;
}

CodeNode* i_check(CodeNode* self)
{
   Trace("i_check");

   assert(code_is_valid(self));

   if (!code_eval_child_bool(self, 0))
   {
      fprintf(stderr, "*** Error 900: Check failed!\n");
      local_print_all(stderr);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   code_value_void(self);

   return self;
}




