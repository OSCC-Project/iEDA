/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: ratsoswrite.c                                                 */
/*   Name....: SOS File Write                                                */
/*   Author..: Thorsten Koch, Daniel Junglas                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2005-2022 by Thorsten Koch <koch@zib.de>
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
#include <ctype.h>
#include <stdbool.h>
#include <assert.h>

#include <gmp.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlp.h"
#include "zimpl/ratlpstore.h"
#include "zimpl/mme.h"

/* Write SOS definitions to a file.
 * A specification for the SOS file format can be found in the
 * ILOG CPLEX Reference Manual.
 */
void lps_sosfile(
   Lps const*  lp,
   FILE*       fp,
   LpFormat    format,
   char const* text)
{
   Sos const*  sos;
   Sse const*  sse;
   int         name_size;
   char*       vtmp;

   assert(lp     != NULL);
   assert(fp     != NULL);
   assert(format == LP_FORM_LPF || format == LP_FORM_MPS);

   name_size = lps_getnamesize(lp, format);
   vtmp      = malloc((size_t)name_size);

   assert(vtmp != NULL);

   if (text != NULL)
      fprintf(fp, "* %s\n", text);
   
   fprintf(fp, "NAME        %8.8s\n", lp->name);
   
   for(sos = lp->sos_root; sos != NULL; sos = sos->next)
   {
      fprintf(fp, "* %s\n", sos->name);

      fprintf(fp, " S%d   %d\n",
         sos->type == SOS_TYPE1 ? 1 : 2,
         sos->priority);
      
      for (sse = sos->first; sse != NULL; sse = sse->next)
      {
         lps_makename(vtmp, name_size, sse->var->name, sse->var->number);
         fprintf(fp, "    %-*s  %.10g\n",
            name_size - 1, vtmp, mpq_get_d(sse->weight));
      }
   }
   fprintf(fp, "ENDATA\n");

   free(vtmp);
}   


