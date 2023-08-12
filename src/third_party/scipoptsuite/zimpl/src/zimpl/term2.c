/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: term2.c                                                        */
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

/* #define TRACE 1 */

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/bound.h"
#include "zimpl/set.h"
#include "zimpl/entry.h"
#include "zimpl/mono.h"
#include "zimpl/hash.h"
#include "zimpl/term.h"
#include "zimpl/stmt.h"
#include "zimpl/prog.h"
#include "zimpl/xlpglue.h"


#define TERM_EXTEND_SIZE 16
#define TERM_SID         0x5465726d

struct term
{
   SID
   Numb*  constant;
   int    size;
   int    used;
   Mono** elem;
};

Term* term_new(int const size)
{
   Term* term = calloc(1, sizeof(*term));

   Trace("term_new");
   
   assert(term != NULL);
   assert(size >  0);
   
   term->constant = numb_new_integer(0);
   term->size     = size;
   term->used     = 0;
   term->elem     = calloc((size_t)term->size, sizeof(*term->elem));
   
   SID_set(term, TERM_SID);
   assert(term_is_valid(term));

   return term;
}
   
void term_add_elem(Term* const term, Entry const* const entry, Numb const* const coeff, MFun const mfun)
{
   Trace("term_add_elem");

   assert(term_is_valid(term));
   assert(entry_is_valid(entry));
   assert(!numb_equal(coeff, numb_zero()));
   assert(term->used <= term->size);
   
   if (term->used == term->size)
   {
      term->size   += TERM_EXTEND_SIZE;
      term->elem    = realloc(
         term->elem, (size_t)term->size * sizeof(*term->elem));

      assert(term->elem != NULL);
   }
   assert(term->used < term->size);

   term->elem[term->used] = mono_new(coeff, entry, mfun);
   term->used++;

   assert(term_is_valid(term));
}

#if 0 /* not used */
void term_mul_elem(Term* term, Entry const* entry, Numb const* coeff)
{
   int i;

   Trace("term_mul_elem");

   assert(term_is_valid(term));
   assert(entry_is_valid(entry));
   assert(!numb_equal(coeff, numb_zero()));
   assert(term->used <= term->size);
   
   for(i = 0; i < term->used; i++)
   {
      mono_mul_entry(term->elem[i], entry);
      mono_mul_coeff(term->elem[i], coeff);
   }
   assert(term_is_valid(term));
}
#endif

void term_free(Term* const term)
{
   Trace("term_free");

   assert(term_is_valid(term));

   SID_del(term);

   for(int i = 0; i < term->used; i++)
      mono_free(term->elem[i]);

   numb_free(term->constant);
   
   free(term->elem);
   free(term);
}

bool term_is_valid(Term const* const term)
{
   if (term == NULL || !SID_ok(term, TERM_SID) || term->used > term->size)
      return false;

   for(int i = 0; i < term->used; i++)
      if (numb_equal(mono_get_coeff(term->elem[i]), numb_zero()))
         return false;      

   return true;
}

Term* term_copy(Term const* const term)
{
   Term* const tnew = term_new(term->used + TERM_EXTEND_SIZE);
   
   Trace("term_copy");
   
   assert(term_is_valid(term));
   assert(term_is_valid(tnew));

   for(int i = 0; i < term->used; i++)
      tnew->elem[i] = mono_copy(term->elem[i]);

   tnew->used = term->used;
   numb_set(tnew->constant, term->constant);

   assert(term_is_valid(tnew));

   return tnew;
}

void term_append_elem(Term* const term, Mono* const mono)
{
   Trace("term_append_elem");

   assert(term_is_valid(term));
   assert(mono_is_valid(mono));

   if (term->used + 1 >= term->size)
   {
      term->size  = term->used + TERM_EXTEND_SIZE;
      term->elem  = realloc(term->elem, (size_t)term->size * sizeof(*term->elem));

      assert(term->elem != NULL);
   }
   assert(term->used < term->size);

   term->elem[term->used] = mono;
   term->used++;

   assert(term_is_valid(term));
}

void term_append_term(
   Term*       const term,
   Term const* const term_b)
{
   /* ??? test auf gleiche monome fehlt!!! */

   Trace("term_append_term");

   assert(term_is_valid(term));
   assert(term_is_valid(term_b));

   if (term->used + term_b->used >= term->size)
   {
      term->size  = term->used + term_b->used;
      term->elem  = realloc(term->elem, (size_t)term->size * sizeof(*term->elem));

      assert(term->elem != NULL);
   }
   assert(term->used + term_b->used <= term->size);

   if (!numb_equal(term_b->constant, numb_zero()))
       numb_add(term->constant, term_b->constant);

   for(int i = 0; i < term_b->used; i++)
   {
      term->elem[term->used] = mono_copy(term_b->elem[i]);
      term->used++;
   }
   assert(term_is_valid(term));
}

Term* term_mul_term(Term const* const term_a, Term const* const term_b)
{
   Trace("term_mul_term");

   assert(term_is_valid(term_a));
   assert(term_is_valid(term_b));

   Term* const term = term_new((term_a->used + 1) * (term_b->used + 1)); /* +1 for constant */

   for(int i = 0; i < term_a->used; i++)
   {
      for(int k = 0; k < term_b->used; k++)
      {
         assert(term->used < term->size);
         
         term->elem[term->used] = mono_mul(term_a->elem[i], term_b->elem[k]);
         term->used++;
      }
   }
   if (!numb_equal(term_b->constant, numb_zero()))
   {
      for(int i = 0; i < term_a->used; i++)
      {
         assert(term->used < term->size);

         term->elem[term->used] = mono_copy(term_a->elem[i]);
         mono_mul_coeff(term->elem[term->used], term_b->constant);
         term->used++;
      }         
   }
   if (!numb_equal(term_a->constant, numb_zero()))
   {
      for(int i = 0; i < term_b->used; i++)
      {
         assert(term->used < term->size);

         term->elem[term->used] = mono_copy(term_b->elem[i]);
         mono_mul_coeff(term->elem[term->used], term_a->constant);
         term->used++;
      }         
   }
   numb_free(term->constant);
   term->constant = numb_new_mul(term_a->constant, term_b->constant);

   assert(term_is_valid(term));
   
   Term* const term_simplified = term_simplify(term);

   term_free(term);

   return term_simplified;
}

Term* term_add_term(Term const* const term_a, Term const* const term_b)
{
   Trace("term_add_term");

   assert(term_is_valid(term_a));
   assert(term_is_valid(term_b));

   Term* const term = term_new(term_a->used + term_b->used + TERM_EXTEND_SIZE);
   term->used       = term_a->used + term_b->used;

   numb_set(term->constant, term_a->constant);
   numb_add(term->constant, term_b->constant);

   assert(term->size >= term->used);

   for(int i = 0; i < term_a->used; i++)
      term->elem[i] = mono_copy(term_a->elem[i]);

   for(int i = 0; i < term_b->used; i++)
      term->elem[i + term_a->used] = mono_copy(term_b->elem[i]);

   assert(term_is_valid(term));

   return term;
}

Term* term_sub_term(Term const* const term_a, Term const* const term_b)
{
   Trace("term_sub_term");
   
   assert(term_is_valid(term_a));
   assert(term_is_valid(term_b));

   Term* const term = term_new(term_a->used + term_b->used + TERM_EXTEND_SIZE);
   term->used       = term_a->used + term_b->used;

   numb_set(term->constant, term_a->constant);
   numb_sub(term->constant, term_b->constant);

   assert(term->size >= term->used);

   for(int i = 0; i < term_a->used; i++)
      term->elem[i] = mono_copy(term_a->elem[i]);

   for(int i = 0; i < term_b->used; i++)
   {
      term->elem[i + term_a->used] = mono_copy(term_b->elem[i]);
      mono_neg(term->elem[i + term_a->used]);
   }
   assert(term_is_valid(term));

   return term;
}

/** Combines monoms in the term where possible
 */  
Term* term_simplify(Term const* const term_org)
{
   assert(term_is_valid(term_org));

   Term* const term = term_new(term_org->used + TERM_EXTEND_SIZE);
   Hash* const hash = hash_new(HASH_MONO, term_org->used);

   numb_set(term->constant, term_org->constant);

   for(int i = 0; i < term_org->used; i++)
   {
      Mono const* mono = hash_lookup_mono(hash, term_org->elem[i]);

      if (mono == NULL)
      {     
         term->elem[term->used] = mono_copy(term_org->elem[i]);

         hash_add_mono(hash, term->elem[term->used]);

         term->used++;
      }
      else
      {
         assert(mono_equal(mono, term_org->elem[i]));
         
         mono_add_coeff(mono, mono_get_coeff(term_org->elem[i]));
      }
   }
   hash_free(hash);
   
   /* Check if there are any cancellations
    */
   for(int i = 0; i < term->used; i++)
   {
      if (numb_equal(mono_get_coeff(term->elem[i]), numb_zero()))
      {
         mono_free(term->elem[i]);

         if (term->used > 0)
         {
            term->used--;
            term->elem[i] = term->elem[term->used];
            i--;
         }
      }
   }
   assert(term_is_valid(term));

   return term;
}

/*lint -e{818} supress "Pointer parameter 'term' could be declared as pointing to const" */
void term_add_constant(
   Term*       const term, 
   Numb const* const value)
{
   Trace("term_add_constant");

   assert(term_is_valid(term));

   numb_add(term->constant, value);

   assert(term_is_valid(term));
}

/*lint -e{818} supress "Pointer parameter 'term' could be declared as pointing to const" */
void term_sub_constant(Term* const term, Numb const* const value)
{
   Trace("term_sub_constant");

   assert(term_is_valid(term));

   numb_sub(term->constant, value);

   assert(term_is_valid(term));
}

void term_mul_coeff(Term* const term, Numb const* const value)
{
   Trace("term_mul_coeff");

   assert(term_is_valid(term));

   numb_mul(term->constant, value);

   if (numb_equal(value, numb_zero()))
   {
      for(int i = 0; i < term->used; i++)
         mono_free(term->elem[i]);
      
      term->used = 0;
   }
   else
   {
      for(int i = 0; i < term->used; i++)
         mono_mul_coeff(term->elem[i], value);
   }
   assert(term_is_valid(term));
}

Numb const* term_get_constant(Term const* const term)
{
   assert(term_is_valid(term));
   
   return term->constant;
}

#if 0 /* not used */
void term_negate(Term* const term)
{
   Trace("term_negate");

   assert(term_is_valid(term));

   term_mul_coeff(term, numb_minusone());
}
#endif

int term_get_elements(Term const* const term)
{
   assert(term_is_valid(term));

   return term->used;
}

Mono* term_get_element(Term const* const term, int const i)
{
   assert(term_is_valid(term));
   assert(i >= 0);
   assert(i <  term->used);
   
   return term->elem[i];
}

Bound* term_get_lower_bound(Term const* const term)
{
   Bound* bound;
   Numb*  lower;
   Numb*  value;
   
   lower = numb_new_integer(0);
      
   for(int i = 0; i < term->used; i++)
   {
      Numb const* const coeff = mono_get_coeff(term->elem[i]);
      int         const sign  = numb_get_sgn(coeff);

      assert(sign != 0);

      bound = (sign > 0)
         ? xlp_getlower(prog_get_lp(), mono_get_var(term->elem[i], 0))
         : xlp_getupper(prog_get_lp(), mono_get_var(term->elem[i], 0));
      
      if (bound_get_type(bound) != BOUND_VALUE)
         goto finish;

      value = numb_new_mul(bound_get_value(bound), coeff);

      numb_add(lower, value);
      
      bound_free(bound);
      numb_free(value);
   }
   numb_add(lower, term->constant);

   bound = bound_new(BOUND_VALUE, lower);

 finish:
   numb_free(lower);

   return bound;
}

Bound* term_get_upper_bound(Term const* const term)
{
   Bound*      bound;
   Numb*       upper;
   Numb*       value;
   
   upper = numb_new_integer(0);
      
   for(int i = 0; i < term->used; i++)
   {
      Numb const* coeff = mono_get_coeff(term->elem[i]);
      int         sign  = numb_get_sgn(coeff);

      assert(sign != 0);

      bound = (sign > 0)
         ? xlp_getupper(prog_get_lp(), mono_get_var(term->elem[i], 0))
         : xlp_getlower(prog_get_lp(), mono_get_var(term->elem[i], 0));
      
      if (bound_get_type(bound) != BOUND_VALUE)
         goto finish;

      value = numb_new_mul(bound_get_value(bound), coeff);

      numb_add(upper, value);
      
      bound_free(bound);
      numb_free(value);
   }
   numb_add(upper, term->constant);
   
   bound = bound_new(BOUND_VALUE, upper);

 finish:
   numb_free(upper);

   return bound;
}

int term_get_degree(Term const* const term)
{
   int degree = 0;
   
   for(int i = 0; i < term->used; i++)
      if (degree < mono_get_degree(term->elem[i]))
         degree = mono_get_degree(term->elem[i]);

   return degree;
}
     
bool term_is_linear(Term const* const term)
{
   for(int i = 0; i < term->used; i++)
      if (!mono_is_linear(term->elem[i]))
         return false;

   return true;
}

bool term_is_polynomial(Term const* const term)
{
   for(int i = 0; i < term->used; i++)
      if (mono_get_function(term->elem[i]) != MFUN_NONE)
         return false;

   return true;
}

bool term_has_realfunction(Term const* const term)
{
   for(int i = 0; i < term->used; i++)
   {
      MFun const fun = mono_get_function(term->elem[i]);
         
      if (fun != MFUN_NONE && fun != MFUN_TRUE && fun != MFUN_FALSE)
         return true;
   }
   return false;
}

Term* term_make_conditional(Term const* const ind_term, Term const* const cond_term, bool const is_true)
{
   assert(term_is_valid(ind_term));
   assert(term_is_valid(cond_term));

   assert(term_get_elements(ind_term) == 1);
   assert(term_get_degree(ind_term)   == 1);
   assert(numb_equal(mono_get_coeff(term_get_element(ind_term, 0)), numb_one()));

   Term* const term = term_copy(ind_term);

   mono_set_function(term_get_element(term, 0), is_true ? MFUN_TRUE : MFUN_FALSE);

   term_append_term(term, cond_term);

   return term;
}

bool term_is_all_integer(Term const* const term)
{
   for(int i = 0; i < term->used; i++)
   {
      VarClass vc = xlp_getclass(prog_get_lp(), mono_get_var(term->elem[i], 0));

      if (vc != VAR_INT && vc != VAR_IMP)
         return false;
   }
   return true;
}

#ifndef NDEBUG
void term_print(FILE* fp, Term const* const term, bool const print_symbol_index)
{
   assert(term_is_valid(term));

   for(int i = 0; i < term->used; i++)
      mono_print(fp, term->elem[i], print_symbol_index);

   if (!numb_equal(term->constant, numb_zero()))
   {
      if (numb_cmp(term->constant, numb_zero()) >= 0)
         fprintf(fp, " + %.16g ", numb_todbl(term->constant));
      else
         fprintf(fp, " - %.16g ", -numb_todbl(term->constant));
   }
}
#endif /* !NDEBUG */







