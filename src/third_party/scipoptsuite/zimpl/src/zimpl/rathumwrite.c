/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: lpfwrite.c                                                    */
/*   Name....: LP Format File Writer                                         */
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

#include "zimpl/attribute.h"
#include "zimpl/gmpmisc.h"
#include "zimpl/ratlp.h"
#include "zimpl/ratlpstore.h"

expects_NONNULL
static void write_name(FILE* fp, char const* name)
{
   char const* s         = name;
   bool        first     = true;
   bool        in_string = false;
   
   assert(fp   != NULL);
   assert(name != NULL);

   /* We allways start with a space
    */
   fputc(' ', fp);

   while(*s != '\0')
   {
      if (*s != '#' && *s != '$')
         fputc(*s, fp);
      else
      {
         if (first)
         {
            first = false;
            fputc('[', fp);
         }
         else
         {
            if (in_string)
            {
               fputc('\"', fp);
               in_string = false;
            }
            fputc(',', fp);
         }
         if (*s == '$')
         {
            assert(!in_string);

            in_string = true;
            fputc('\"', fp);
         }
      }
      s++;
   }
   if (in_string)
      fputc('\"', fp);
   if (!first)
      fputc(']', fp);
}

expects_NONNULL
static void write_lhs(FILE* fp, Con const* con, ConType type)
{
   assert(fp  != NULL);
   assert(con != NULL);
   
   switch(type)
   {
   case CON_RHS :
   case CON_LHS :
   case CON_EQUAL :
      break;
   case CON_RANGE :
      mpq_out_str(fp, 10, con->lhs);
      fprintf(fp, " <= ");
      break;
   default :
      abort();
   }
}

expects_NONNULL
static void write_rhs(FILE* fp, Con const* con, ConType type)
{
   assert(fp  != NULL);
   assert(con != NULL);
   
   switch(type)
   {
   case CON_RHS :
   case CON_RANGE :
      fprintf(fp, " <= ");
      mpq_out_str(fp, 10, con->rhs);
      break;
   case CON_LHS :
      fprintf(fp, " >= ");
      mpq_out_str(fp, 10, con->lhs);
      break;
   case CON_EQUAL :
      fprintf(fp, " = ");
      mpq_out_str(fp, 10, con->rhs);
      break;
   default :
      abort();
   }
   fprintf(fp, "\n");
}

expects_NONNULL
static void write_row(
   FILE* fp,
   Con const* con)
{
   Nzo const* nzo;
   int        cnt = 0;

   assert(fp   != NULL);
   assert(con  != NULL);
   
   for(nzo = con->first; nzo != NULL; nzo = nzo->con_next)
   {
      if (mpq_equal(nzo->value, const_one))
         fprintf(fp, " +");
      else if (mpq_equal(nzo->value, const_minus_one))
         fprintf(fp, " -");
      else
      {
         fputc(' ', fp);
         
         /*lint -e(634) Strong type mismatch (type 'Bool') in equality or conditional */
         if (mpq_sgn(nzo->value) > 0)
            fprintf(fp, "+");
         
         mpq_out_str(fp, 10, nzo->value);
      }
      write_name(fp, nzo->var->name);
      
      if (++cnt % 6 == 0)
         fprintf(fp, "\n ");         
   }
}

expects_NONNULL12
void hum_write(
   Lps const*  lp,
   FILE*       fp,
   char const* text)
{
   Var const* var;
   Con const* con;
   bool  have_integer  = false;
   int   cnt;
   
   assert(lp       != NULL);
   assert(fp       != NULL);

   if (text != NULL)
      fprintf(fp, "%s\n", text);
   
   fprintf(fp, "Problem: %s\n", lp->name);   
   fprintf(fp, "%s\n", (lp->direct == LP_MIN) ? "Minimize" : "Maximize");
   fprintf(fp, " %s: ", lp->objname == NULL ? "Objective" : lp->objname);
   
   for(var = lp->var_root, cnt = 0; var != NULL; var = var->next)
   {
      /* If cost is zero, do not include in objective function
       */
      if (mpq_equal(var->cost, const_zero))
         continue;

      if (mpq_equal(var->cost, const_one))
         fprintf(fp, " +");
      else if (mpq_equal(var->cost, const_minus_one))
         fprintf(fp, " -");
      else
      {
         fputc(' ', fp);
         
         /*lint -e(634) Strong type mismatch (type 'Bool') in equality or conditional */
         if (mpq_sgn(var->cost) > 0)
            fprintf(fp, "+");
         
         mpq_out_str(fp, 10, var->cost);
      }
      write_name(fp, var->name);

      if (++cnt % 6 == 0)
         fprintf(fp, "\n ");
   }
   /* ---------------------------------------------------------------------- */

   fprintf(fp, "\nSubject to:\n");

   for(con = lp->con_root; con != NULL; con = con->next)
   {
      if (con->size == 0)
         continue;

      write_name(fp, con->name);
      fprintf(fp, ":\n ");

      write_lhs(fp, con, con->type);
      write_row(fp, con);
      write_rhs(fp, con, con->type);
   }

   /* ---------------------------------------------------------------------- */

   fprintf(fp, "Bounds\n");

   for(var = lp->var_root; var != NULL; var = var->next)
   {
      /* A variable without any entries in the matrix
       * or the objective function can be ignored.
       */
      if (var->size == 0 && mpq_equal(var->cost, const_zero) && !lps_has_sos(lp))
         continue;

      if (var->type == VAR_FIXED)
      {
         write_name(fp, var->name);
         fprintf(fp, " = ");
         mpq_out_str(fp, 10, var->lower);
      }
      else
      {
         /* Check if we have integer variables
          */
         if (var->vclass == VAR_INT)
            have_integer = true;
         
         if (var->type == VAR_LOWER || var->type == VAR_BOXED)
            mpq_out_str(fp, 10, var->lower);
         else
            fprintf(fp, " -Inf");
         
         fprintf(fp, " <=");
         write_name(fp, var->name);
         fprintf(fp, " <= ");
         
         if (var->type == VAR_UPPER || var->type == VAR_BOXED)
            mpq_out_str(fp, 10, var->upper);
         else
            fprintf(fp, "+Inf");
      }
      fprintf(fp, "\n");
   }

   /* ---------------------------------------------------------------------- */

   if (have_integer)
   {
      fprintf(fp, "General\n");
      
      for(var = lp->var_root; var != NULL; var = var->next)
      {
         if (var->vclass != VAR_INT)
            continue;

         if (var->size == 0 && mpq_equal(var->cost, const_zero) && !lps_has_sos(lp))
            continue;
         
         write_name(fp, var->name);
         fputc('\n', fp);
      }
   }

   /* ---------------------------------------------------------------------- */

   if (lps_has_sos(lp))
   {
      Sos const* sos;
      Sse const* sse;

      fprintf(fp, "SOS\n");

      for(sos = lp->sos_root; sos != NULL; sos = sos->next)
      {
         fprintf(fp, "%s S%d ", 
            sos->name,
            sos->type == SOS_TYPE1 ? 1 : 2);

         for(sse = sos->first; sse != NULL; sse = sse->next)
         {
            write_name(fp, sse->var->name);
            mpq_out_str(fp, 10, sse->weight);
            fputc('\n', fp);
         }
      }
   }

   fprintf(fp, "End\n");

}   

/* ------------------------------------------------------------------------- */
/* Emacs Local Variables:                                                    */
/* Emacs mode:c                                                              */
/* Emacs c-basic-offset:3                                                    */
/* Emacs tab-width:8                                                         */
/* Emacs indent-tabs-mode:nil                                                */
/* Emacs End:                                                                */
/* ------------------------------------------------------------------------- */
