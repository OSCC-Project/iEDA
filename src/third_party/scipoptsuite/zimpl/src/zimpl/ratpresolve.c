/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: lpstore.c                                                     */
/*   Name....: Store Linear Programm                                         */
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
#if 0 /* Not used anymore ??? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <gmp.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"
#include <stdbool.h>
#include "zimpl/gmpmisc.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/bound.h"
#include "zimpl/mme.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/ratlp.h"
#include "zimpl/ratlpstore.h"

/*lint -e{818} supress "Pointer parameter 'var' could be declared as pointing to const" */
static void remove_fixed_var(
   Lps* lp,
   Var* var,
   int  verbose_level)
{
   Nzo*  nzo;
   mpq_t x;
   mpq_t temp;
   bool  is_zero;
   
   assert(lp  != NULL);
   assert(var != NULL);
   
   assert(var->type == VAR_FIXED && mpq_equal(var->lower, var->upper));

   if (verbose_level > 0)
      printf("Removing variable %s fixed to %g\n", var->name, mpq_get_d(var->lower));
   
   mpq_init(x);
   mpq_init(temp);

   is_zero = mpq_equal(var->lower, x /* zero */);

   while(var->first != NULL)
   {
      nzo = var->first;

      /* Do we have to ajust the lhs/rhs ?
       */
      if (!is_zero)
      {
         mpq_mul(x, var->lower, nzo->value);

         /* do we have a valid rhs ?
          */
         if (HAS_RHS(nzo->con))
         {
            mpq_sub(temp, nzo->con->rhs, x);
            lps_setrhs(nzo->con, temp);
         }
         /* do we have a valid lhs ?
          */
         if (HAS_LHS(nzo->con))
         {
            mpq_sub(temp, nzo->con->lhs, x);
            lps_setlhs(nzo->con, temp);
         }
      }
      lps_delnzo(lp, nzo);
   }
   mpq_clear(temp);
   mpq_clear(x);
}

static PSResult simple_rows(
   Lps*  lp,
   Bool* again,
   int   verbose_level)
{
   bool  have_up;
   bool  have_lo;
   mpq_t up;
   mpq_t lo;
   Nzo*  nzo;
   Var*  var;
   Con*  con;
   Con*  con_next;
   int   rem_rows = 0;
   int   rem_nzos = 0;
   
   mpq_init(up);
   mpq_init(lo);
   
   for(con = lp->con_root; con != NULL; con = con_next)
   {
      con_next = con->next;
      
      /* infeasible range row
       */
      if (con->type == CON_RANGE && mpq_cmp(con->lhs, con->rhs) > 0)
      {
         printf("Infeasible range row %s lhs=%g rhs=%g\n",
            con->name, mpq_get_d(con->lhs), mpq_get_d(con->rhs));

         return PRESOLVE_INFEASIBLE;
      }
      /* empty row ?
       */
      if (con->size == 0)
      {
         if (  (HAS_RHS(con) && mpq_cmp(con->rhs, const_zero) < 0)
            || (HAS_LHS(con) && mpq_cmp(con->lhs, const_zero) > 0))
         {
            printf("Infeasible empty row %s lhs=%g rhs=%g\n", 
               con->name, mpq_get_d(con->lhs), mpq_get_d(con->rhs));

            return PRESOLVE_INFEASIBLE;
         }         
         if (verbose_level > 0)
            printf("Empty row %s removed\n", con->name);

         lps_delcon(lp, con);
         
         rem_rows++;
         continue;
      }
      /* unconstraint constraint
       */
      if (con->type == CON_FREE)
      {
         if (verbose_level > 0)
            printf("Unconstraint row %s removed\n", con->name);

         rem_rows++;
         rem_nzos += con->size;

         lps_delcon(lp, con);
         continue;
      }
      /* row singleton
       */
      if (con->size == 1)
      {
         have_up = false;
         have_lo = false;
         nzo     = con->first;
         var     = nzo->var;

         if (mpq_cmp(nzo->value, const_zero) > 0) /* x > 0 */
         {
            if (HAS_RHS(con)) 
            {
               mpq_div(up, con->rhs, nzo->value);
               have_up = true;
            }
            if (HAS_LHS(con))
            {
               mpq_div(lo, con->lhs, nzo->value);
               have_lo = true;
            }
         }
         else if (mpq_cmp(nzo->value, const_zero) < 0) /* x < 0 */
         {
            if (HAS_RHS(con))
            {
               mpq_div(lo, con->rhs, nzo->value);
               have_lo = true;
            }
            if (HAS_LHS(con))
            {
               mpq_div(up, con->lhs, nzo->value);
               have_up = true;
            }
         }
         else if ((HAS_RHS(con) && !mpq_equal(con->rhs, const_zero))
            ||    (HAS_LHS(con) && !mpq_equal(con->lhs, const_zero)))  
         {
            /* x == 0 rhs/lhs != 0 */
            printf("Infeasibel row %s Zero row singleton with non zero lhs or rhs\n",
               con->name);
            
            return PRESOLVE_INFEASIBLE;
         }
         assert(!HAS_LOWER(var) || !HAS_UPPER(var) || mpq_cmp(var->lower, var->upper) <= 0);

         if (have_up && (!HAS_UPPER(var) || mpq_cmp(up, var->upper) < 0))
            lps_setupper(var, up);

         if (have_lo && (!HAS_LOWER(var) || mpq_cmp(lo, var->lower) > 0))
            lps_setlower(var, lo);

         if (HAS_LOWER(var) && HAS_UPPER(var) && mpq_cmp(var->lower, var->upper) > 0)
         {
            printf("Row %s implise infeasible bounds on var %s, lower=%g upper=%g\n",
               con->name, var->name, mpq_get_d(var->lower), mpq_get_d(var->upper));

            return PRESOLVE_INFEASIBLE;
         }

         if (verbose_level > 1)
            printf("Row %s singleton var %s set to lower=%g upper=%g\n",
               con->name, var->name, mpq_get_d(var->lower), mpq_get_d(var->upper));
         
         rem_rows++;
         rem_nzos++;

         lps_delcon(lp, con);
         continue;
      }
   }
   assert(rem_rows != 0 || rem_nzos == 0);

   if (rem_rows > 0)
   {
      *again = true;

      if (verbose_level > 0)
         printf("Simple row presolve removed %d rows and %d non-zeros\n",
            rem_rows, rem_nzos);
   }

   mpq_clear(up);
   mpq_clear(lo);
   
   return PRESOLVE_OKAY;
}

static PSResult handle_col_singleton(
   Lps* lp,
   Var* var,
   int  verbose_level,
   int* rem_cols,
   int* rem_nzos)
{
   Nzo*  nzo;
   Con*  con;
   int   cmpres;
   mpq_t maxobj;
   
   assert(lp            != NULL);
   assert(var           != NULL);
   assert(verbose_level >= 0);
   assert(rem_cols      != NULL);
   assert(rem_nzos      != NULL);
   
   nzo = var->first;
   con = nzo->con;
         
   assert(!mpq_equal(nzo->value, const_zero));

   mpq_init(maxobj);
   mpq_set(maxobj, var->cost);
         
   if (lp->direct == LP_MIN)
      mpq_neg(maxobj, maxobj);

   /* Value is positive
    */
   if (mpq_cmp(nzo->value, const_zero) > 0.0)
   {
      /* max -3 x
       * s.t. 5 x (<= 8)
       * l <= x
       *
       * and we have NO lhs, (rhs does not matter)
       */
      if (!HAS_LHS(con))
      {
         cmpres = mpq_cmp(maxobj, const_zero);

         /* The objective is either zero or negative
          */
         if (cmpres <= 0)
         {
            /* If we have no lower bound
             */
            if (!HAS_LOWER(var))
            {
               /* ... but a negative objective, so we get unbounded
                */
               if (cmpres < 0)
               {
                  printf("Unbounded var %s\n", var->name);
                  return PRESOLVE_UNBOUNDED;
               }
               /* With a zero objective and no bounds, there is not
                * much we can do.
                */
            }
            else
            {
               /* now we know we want to go to the lower bound.
                */
               lps_setupper(var, var->lower);
               
               remove_fixed_var(lp, var, verbose_level);
               
               (*rem_cols)++;
               (*rem_nzos)++;

               return PRESOLVE_OKAY;
            }
         }
      }
      /* max  3 x
       * s.t. 5 x (>= 8)
       * x <= u
       *
       * and we have NO rhs, (lhs does not matter)
       */
      if (!HAS_RHS(con))
      {
         cmpres = mpq_cmp(maxobj, const_zero);

         /* The objective is either zero or positive
          */
         if (cmpres >= 0)
         {
            /* If we have no upper bound
             */
            if (!HAS_UPPER(var))
            {
               /* ... but a positive objective, so we get unbounded
                */
               if (cmpres > 0)
               {
                  printf("Unbounded var %s\n", var->name);
                  return PRESOLVE_UNBOUNDED;
               }
               /* With a zero objective and no bounds, there is not
                * much we can do.
                */
            }
            else
            {
               /* now we know we want to go to the upper bound.
                */
               lps_setlower(var, var->upper);

               remove_fixed_var(lp, var, verbose_level);

               (*rem_cols)++;
               (*rem_nzos)++;

               return PRESOLVE_OKAY;
            }
         }
      }
   }
   else
   {
      /* Value is negative
       */
      assert(mpq_cmp(nzo->value, const_zero) < 0.0);

      /* max  -3 x
       * s.t. -5 x (>= 8)
       * l <= x
       */
      if (!HAS_RHS(con))
      {
         cmpres = mpq_cmp(maxobj, const_zero);

         if (cmpres <= 0)
         {
            if (!HAS_LOWER(var))
            {
               if (cmpres < 0)
               {
                  printf("Unbounded var %s\n", var->name);
                  return PRESOLVE_UNBOUNDED;
               }
            }
            else
            {
               lps_setupper(var, var->lower);

               remove_fixed_var(lp, var, verbose_level);
               
               (*rem_cols)++;
               (*rem_nzos)++;               

               return PRESOLVE_OKAY;
            }
         }
      }
      /* max  3 x
       * s.t. -5 x (<= 8)
       * x <= u
       */
      if (!HAS_LHS(con))
      {
         cmpres = mpq_cmp(maxobj, const_zero);

         if (cmpres >= 0)
         {
            if (!HAS_UPPER(var))
            {
               if (cmpres > 0)
               {
                  printf("Unbounded var %s\n", var->name);
                  return PRESOLVE_UNBOUNDED;
               }
            }
            else
            {
               lps_setlower(var, var->upper);

               remove_fixed_var(lp, var, verbose_level);

               (*rem_cols)++;
               (*rem_nzos)++;

               return PRESOLVE_OKAY;
            }
         }
      }
   }
   mpq_clear(maxobj);
   
   return PRESOLVE_OKAY;
}

static PSResult simple_cols(
   Lps*  lp,
   Bool* again,
   int   verbose_level)
{
   PSResult res;
   mpq_t    maxobj;
   int      rem_cols = 0;
   int      rem_nzos = 0;
   Var*     var;

   mpq_init(maxobj);
   
   for(var = lp->var_root; var != NULL; var = var->next)
   {
      if (var->type == VAR_FIXED && var->size == 0)
         continue;
      
      /* Empty column ?
       */
      if (var->size == 0)
      {
         mpq_set(maxobj, var->cost);
         
         if (lp->direct == LP_MIN)
            mpq_neg(maxobj, maxobj);
         
         if (mpq_cmp(maxobj, const_zero) > 0)
         {
            /* Do we not have an upper bound ?
             */
            if (!HAS_UPPER(var))
            {
               printf("Var %s unbounded\n", var->name);

               return PRESOLVE_UNBOUNDED;
            }
            lps_setlower(var, var->upper);
         }
         else if (mpq_cmp(maxobj, const_zero) < 0)
         {
            /* Do we not have an lower bound ?
             */
            if (!HAS_LOWER(var))
            {
               printf("Var %s unbounded\n", var->name);

               return PRESOLVE_UNBOUNDED;
            }
            lps_setupper(var, var->lower);
         }
         else 
         {
            assert(mpq_equal(maxobj, const_zero));
            
            /* any value within the bounds is ok
             */
            if (HAS_LOWER(var))
               lps_setupper(var, var->lower);
            else if (HAS_UPPER(var))
               lps_setlower(var, var->upper);
            else
            {
               lps_setlower(var, const_zero);
               lps_setupper(var, const_zero);
            }
         }
         if (verbose_level > 1)
            printf("Empty var %s fixed\n", var->name);
         
         rem_cols++;
         continue;
      }

      /* infeasible bounds ?
       */
      if (var->type == VAR_BOXED && mpq_cmp(var->lower, var->upper) > 0)
      {
         printf("Var %s infeasible bounds lower=%g upper=%g\n",
            var->name, mpq_get_d(var->lower), mpq_get_d(var->upper));

         return PRESOLVE_INFEASIBLE;
      }
      /* Fixed column ?
       */
      if (var->type == VAR_FIXED)
      {
         assert(var->size > 0);

         rem_cols++;
         rem_nzos += var->size;

         remove_fixed_var(lp, var, verbose_level);

         continue;
      }
      /* Column singleton
       */
      if (var->size == 1)
      {
         res = handle_col_singleton(lp, var, verbose_level, &rem_cols, &rem_nzos);

         if (res != PRESOLVE_OKAY)
            return res;
      }
   }
   assert(rem_cols != 0 || rem_nzos == 0);

   if (rem_cols > 0)
   {
      *again = true;

      if (verbose_level > 0)
         printf("Simple col presolve removed %d cols and %d non-zeros\n",
            rem_cols, rem_nzos);
   }

   mpq_clear(maxobj);
   
   return PRESOLVE_OKAY;
}



PSResult lps_presolve(Lps* lp, int verbose_level)
{
   PSResult ret = PRESOLVE_OKAY;
   bool     again;
   /*
   bool     rcagain;
   bool     rragain;
   */
   do
   {
      again = false;
 
      if (ret == PRESOLVE_OKAY)
         ret = simple_rows(lp, &again, verbose_level);

      if (ret == PRESOLVE_OKAY)
         ret = simple_cols(lp, &again, verbose_level);

      assert(ret == PRESOLVE_OKAY || again == false);
   }
   while(again);

#if 0
   if (ret == OKAY)
      ret = redundantCols(lp, rcagain);

   if (ret == OKAY)
      ret = redundantRows(lp, rragain);

   again = (ret == OKAY) && (rcagain || rragain);

   /* This has to be a loop, otherwise we could end up with
    * empty rows.
    */
   while(again)
   {
      again = false;

      if (ret == OKAY)
         ret = simpleRows(lp, again);

      if (ret == OKAY)
         ret = simpleCols(lp, again);

      assert(ret == OKAY || again == false);
   }
   VERBOSE1({ std::cout << "IREDSM25 redundant simplifier removed "
                        << m_remRows << " rows, "
                        << m_remNzos << " nzos, changed "
                        << m_chgBnds << " col bounds " 
                        << m_chgLRhs << " row bounds,"
                        << std::endl; });

#endif
   /* ??? does not work, because vars are not deleted */
   if (lp->vars == 0 || lp->cons == 0)
   {
      /* ??? Check if this is ok.
       */
      assert(lp->vars == 0 && lp->cons == 0);
      
      printf("Simplifier removed all variables and constraints\n");
      ret = PRESOLVE_VANISHED;
   }
   return ret;
}

#endif

