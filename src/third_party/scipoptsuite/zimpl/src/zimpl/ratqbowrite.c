/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: qbowrite.c                                                    */
/*   Name....: QUBO Format File Writer                                       */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2003-2022 by Thorsten Koch <koch@zib.de>
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

#include <gmp.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/gmpmisc.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/bound.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/entry.h"
#include "zimpl/ratlp.h"
#include "zimpl/ratlpstore.h"
#include "zimpl/random.h"

struct qubo
{
   int    rows;
   int    size;
   int    used;
   int*   rowbeg;
   int*   col;
   mpq_t* val;
};

typedef struct qubo Qubo;

struct qmentry
{
   unsigned int sid;
   Var const*   var1;
   Var const*   var2;
   mpq_t        value;
};

#define QME_SID 0x514D656E

typedef struct qmentry Qme;

is_MALLOC returns_NONNULL
static Qubo *qubo_new(
   int const rows,
   int const nonzeros)
{
   assert(rows     > 0);
   assert(nonzeros > 0);

   Qubo* const qubo = calloc(1, sizeof(*qubo));

   qubo->rows   = rows;
   qubo->size   = nonzeros;

   qubo->rowbeg = calloc((size_t)rows + 1, sizeof(*qubo->rowbeg));
   qubo->col    = calloc((size_t)nonzeros, sizeof(*qubo->col));
   qubo->val    = calloc((size_t)nonzeros, sizeof(*qubo->val));

   for(int i = 0; i < nonzeros; i++)
      mpq_init(qubo->val[i]);

   return qubo;
}

expects_NONNULL
static void qubo_free(Qubo* const qubo)
{
   assert(qubo);

   for(int i = 0; i < qubo->size; i++)
      mpq_clear(qubo->val[i]);
   
   free(qubo->rowbeg);
   free(qubo->col);
   free(qubo->val);
   free(qubo);
}

expects_NONNULL
static int qme_cmp_row(void const* const a, void const* const b)
{
   Qme const* const aa = (Qme const*)a;
   Qme const* const bb = (Qme const*)b;

   assert(aa->sid == QME_SID);
   assert(bb->sid == QME_SID);
   
   int d = aa->var1->number - bb->var1->number;

   if (0 == d)
      d = aa->var2->number - bb->var2->number;

   return d;
}

expects_NONNULL returns_NONNULL
static Qubo* qubo_from_entries(int rows, int qme_used, Qme* qme)
{
   qsort(qme, (size_t)qme_used, sizeof(*qme), qme_cmp_row);

#ifndef NDEBUG
   for(int i = 1; i < qme_used; i++)
      assert(qme[i - 1].var1->number < qme[i].var1->number || (qme[i - 1].var1->number == qme[i].var1->number && qme[i - 1].var2->number <= qme[i].var2->number));   

   for(int i = 0; i < qme_used; i++)
   {
      assert(!mpq_equal(qme[i].value, const_zero));

      if (i + 1 < qme_used && qme[i].var1->number == qme[i + 1].var1->number && qme[i].var2->number == qme[i + 1].var2->number)
         assert(!mpq_equal(qme[i].value, qme[i + 1].value)); // non-symmetric matrix
   }
#endif

   Qubo* const qubo = qubo_new(rows, qme_used);

   int prev_row = -1;
   
   for(int i = 0; i < qme_used; i++)
   {      
      assert(!mpq_equal(qme[i].value, const_zero));

      int const row = qme[i].var1->number;

      if (prev_row != row)
      {
         for(int k = prev_row + 1; k <= row; k++)
            qubo->rowbeg[k] = qubo->used;
         prev_row = row;
      }
      assert(prev_row < rows);

      qubo->col [qubo->used] = qme[i].var2->number;

      mpq_set(qubo->val[qubo->used], qme[i].value);
      
      qubo->used++;
   }
   for(int k = prev_row + 1; k <= rows; k++)
      qubo->rowbeg[k] = qubo->used;

   assert(qme_used == qubo->used);

   return qubo;      
}

expects_NONNULL
static Var const* find_offsetvar(Lps const* const lp)
{
   char const* const format = "%sObjOffset";
   char      * const vname  = malloc(strlen(SYMBOL_NAME_INTERNAL) + strlen(format) + 1);

   sprintf(vname, format, SYMBOL_NAME_INTERNAL);

   Var const* const var_offset = lps_getvar(lp, vname);

   free(vname);

   return var_offset;
}

/* A specification for the QUBO LP file format can be found in the
 */
void qbo_write(
   Lps const*  const lp,
   FILE*       const fp,
   LpFormat    const format,
   char const* const format_options,
   char const* const text)
{
   assert(lp != NULL);
   assert(fp != NULL);

   if (lp->obj_term != NULL && term_get_degree(lp->obj_term) > 2)
   {
      fprintf(stderr, "--- Warning 602: QUBO file format can only handle linear and quadratic terms\n");
      fprintf(fp, "%s0 0 0\n", strchr(format_options, 'p') == NULL ? "" : "p ");
      return;
   }

   if (text != NULL)
      fprintf(fp, "%s", text);   

   /* Add linear part to term
    */
   Var const* const offset_var = find_offsetvar(lp);
   Term     * const term_seq   = term_new(lp->vars + ((lp->obj_term != NULL) ? term_get_elements(lp->obj_term) : 0));
   Tuple    * const tuple      = tuple_new(0);

   for(Var* var = lp->var_root; var != NULL; var = var->next)
   {
      if (var == offset_var)
         continue;

      if (mpq_equal(var->cost, const_zero))
         continue;

      if (!lps_is_binary(var))
      {
         if (verbose > 0)
         {
            fprintf(stderr, "--- Warning 601: File format can only handle binary variables\n");
            fprintf(stderr, "                 Non-binary variable \"%s\" ignored\n", 
               var->name);
         }
         continue;
      }
      Entry* const entry = entry_new_var(tuple, var);
      Numb*  const cost  = numb_new_mpq(var->cost);
      Mono*  const mono  = mono_new(cost, entry, MFUN_NONE);      
      
      //old mono_mul_entry(mono, entry); // necessary such that simplify works
      term_append_elem(term_seq, mono);
      entry_free(entry);
      numb_free(cost);
   }
   /* Now add qudratic part and make into a list
    */
   if (lp->obj_term != NULL)
   {
      //old term_append_term(term_seq, lp->obj_term);

      // Since multiplying a binary variable with itsself is useless,
      // we remove these. This is neccessary for simplify to work.
      for(int i = 0; i < term_get_elements(lp->obj_term); i++)
      {
         Mono const* const mono     = term_get_element(lp->obj_term, i);
         Mono*             new_mono = NULL;
         int         const degree   = mono_get_degree(mono);

         if (degree == 1)
            new_mono = mono_copy(mono);
         else
         {
            assert(degree == 2);

            Var* const var = mono_get_var(mono, 0);

            assert(lps_is_binary(var));
      
            if (var != mono_get_var(mono, 1))
               new_mono = mono_copy(mono);
            else
            {
               // now we have the case that its x^2 with x binary.
               Entry*      const entry = entry_new_var(tuple, var);
               Numb const* const coeff = mono_get_coeff(mono);

               new_mono = mono_new(coeff, entry, MFUN_NONE);
               entry_free(entry);
            }
         }
         assert(new_mono != NULL);
         
         term_append_elem(term_seq, new_mono);
      }
   }
   tuple_free(tuple);

   Term* const term_obj = term_simplify(term_seq);

   term_free(term_seq);

   // Put into Q
   int  const qme_size = term_get_elements(term_obj);

   if (qme_size == 0)
   {
      fprintf(stderr, "--- Warning 603: QUBO instance empty\n");
      fprintf(fp, "%s0 0 0\n", strchr(format_options, 'p') == NULL ? "" : "p ");
      return;
   }
   Qme* const qme      = calloc((size_t)qme_size, sizeof(*qme));
   int        qme_used = 0;
   mpq_t      offset;

   mpq_init(offset);

   numb_get_mpq(term_get_constant(term_obj), offset);

   if (offset_var != NULL)
      mpq_add(offset, offset, offset_var->cost);
         
   // TODO maybe move this into QUBO from qme.
   for(int i = 0; i < qme_size; i++)
   {
      Mono const* const mono = term_get_element(term_obj, i);

      int const degree = mono_get_degree(mono);

      assert(degree == 1 || degree == 2);
      
      Var const* const var1 = mono_get_var(mono, 0);
      Var const* const var2 = (degree == 1) ? var1 : mono_get_var(mono, 1);

      // Check here quadratic
      qme[qme_used].sid  = QME_SID;

      qme[qme_used].var1 = var1->number <= var2->number ? var1 : var2;
      qme[qme_used].var2 = var1->number <= var2->number ? var2 : var1;

      assert(qme[qme_used].var1->number <= qme[qme_used].var2->number);

      mpq_init(qme[qme_used].value);

      numb_get_mpq(mono_get_coeff(mono), qme[qme_used].value);

      assert(!mpq_equal(qme[qme_used].value, const_zero));

      qme_used++;         
   }
   assert(qme_used == qme_size);

   term_free(term_obj);
   
   Qubo* const qubo = qubo_from_entries(lp->vars, qme_used, qme);

   for(int i = 0; i < qme_used; i++)
      mpq_clear(qme[i].value);
   
   free(qme);

   char const* const start_comment = (strchr(format_options, 'c')) == NULL ? "#" : "c";
   
   fprintf(fp, "%s ObjectiveOffset %ld\n",
      start_comment, 
      (long)mpq_get_d(offset));

#ifdef TO_BE_IMPLEMENTED
   fprintf(fp, "%s FeasibilityInstance %s\n",
      start_comment, 
      );
   
   fprintf(fp, "%s SolutionCardinality %d\n",
      start_comment, 
      );
#endif 
   fprintf(fp, "%s Vars Non-zeros\n", start_comment);

   // if offset_var is the last variable, reduce the number of vars given by 1
   // Overall it can happen that variables which are no constraint are counted here.   
   fprintf(fp, "%s%d %d\n",
      strchr(format_options, 'p') == NULL ? "" : "p ",
      lp->vars - ((offset_var != NULL && (offset_var->number == lp->vars - 1)) ? 1 : 0),
      qme_used);
   
   int    const index_base = (strchr(format_options, '0') == NULL) ? 1 : 0;
   double const sign       = (lp->direct == LP_MIN) ? 1.0 : -1.0;
   
   for(int row = 0; row < qubo->rows; row++)
   {
      for(int k = qubo->rowbeg[row]; k < qubo->rowbeg[row + 1]; k++)
      {
         int col = qubo->col[k];

         assert(row <= col);
         
         // divide off diagonal entries by two as they will be doubled later
         // this is the biqmac format
         fprintf(fp, "%d %d %.15g\n",
            row + index_base,
            col + index_base,
            ((row == col) ? 1.0 : 0.5) * sign * mpq_get_d(qubo->val[k]));
      }
   }
   mpq_clear(offset);   
   qubo_free(qubo);
}   




/* ------------------------------------------------------------------------- */
/* Emacs Local Variables:                                                    */
/* Emacs mode:c                                                              */
/* Emacs c-basic-offset:3                                                    */
/* Emacs tab-width:8                                                         */
/* Emacs indent-tabs-mode:nil                                                */
/* Emacs End:                                                                */
/* ------------------------------------------------------------------------- */
