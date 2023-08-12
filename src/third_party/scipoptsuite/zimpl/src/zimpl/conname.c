/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: conname.c                                                     */
/*   Name....: Constraint Names                                              */
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

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/mme.h"
#include "zimpl/bound.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/local.h"
#include "zimpl/stmt.h"
#include "zimpl/prog.h"
#include "zimpl/xlpglue.h"
#include "zimpl/conname.h"

static char*        cpfix  = NULL;
static int          count  = 1;
static char*        cname  = NULL;
static size_t       clen   = 0;
static ConNameForm  cform  = CON_FORM_NAME;

void conname_format(ConNameForm format)
{
   cform = format;
}

void conname_free()
{
   assert(cname != NULL);
   assert(cpfix != NULL);

   free(cname);
   free(cpfix);
   
   cname = NULL;
   clen  = 0;
}

/* return False if we are in mode CON_FORM_NAME and
 * already a constraint with the prefix exists. Otherwise this
 * is unimportant, because all constraints will get a unique
 * number anyway.
 */
bool conname_set(char const* prefix)
{
   assert(prefix != NULL);
   assert(cname  == NULL);

   cpfix = strdup(prefix);
   clen  = strlen(cpfix) + 16;
   cname = malloc(clen);

   assert(cname != NULL);

   if (cform != CON_FORM_NAME)
      return true;

   assert(cform == CON_FORM_NAME);
   
   count = 1;
   
   strcpy(cname, cpfix);
   strcat(cname, "_1");

   if (xlp_conname_exists(prog_get_lp(), cname))
      return false;

   strcat(cname, "_a_0");

   return !xlp_conname_exists(prog_get_lp(), cname);
}

char const* conname_get()
{
   char*  localstr;
   size_t newlen;
   
   assert(cpfix != NULL);
   assert(cname != NULL);

   switch(cform)
   {
   case CON_FORM_MAKE :
      snprintf(cname, clen, "c%d", count);
      break;
   case CON_FORM_NAME :
      snprintf(cname, clen, "%s_%d", cpfix, count);
      break;
   case CON_FORM_FULL :
      localstr = local_tostrall();
      newlen   = strlen(localstr) + strlen(cpfix) + 16;

      if (newlen > clen)
      {
         clen  = newlen;
         cname = realloc(cname, clen);

         assert(cname != NULL);
      }
      snprintf(cname, clen, "%s_%s%s",
         cpfix,
         strlen(localstr) > 0 ? ";" : "",
         localstr);

      free(localstr);
      break;
   }
   assert(strlen(cname) < clen);
   
   return cname;
}

void conname_next()
{
   count++;
}

