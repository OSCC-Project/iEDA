/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: rdefpar.c                                                     */
/*   Name....: Read Definition / Parameter                                   */
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

#include "zimpl/mme.h"
#include "zimpl/strstore.h"
#include "zimpl/rdefpar.h"

#define RDEF_SID     0x52446566
#define RPAR_SID     0x52506172

enum read_param_type { RPAR_ERR = 0, RPAR_SKIP, RPAR_USE, RPAR_CMNT, RPAR_MTCH };

typedef enum read_param_type   RParType;
typedef union read_param_value RParVal;

union read_param_value
{
   int         i;
   char const* s;
};

struct read_param
{
   SID
   RParType type;
   RParVal  val;
};
   
struct read_definition
{
   SID
   char const* filename;
   char const* pattern;  /* this was named "template", but template */   
   char const* comment;  /* is a C++ reserved word */
   char const* match;
   int         use;
   int         skip;
   int         refc;
};

RDef* rdef_new(char const* filename, char const* pattern)
{
   RDef* rdef = calloc(1, sizeof(*rdef));

   assert(filename != NULL);
   assert(pattern  != NULL);
   assert(rdef     != NULL);
   
   rdef->filename = filename;
   rdef->pattern  = pattern;
   rdef->comment  = str_new("");
   rdef->match    = NULL;
   rdef->skip     = 0;
   rdef->use      = -1;
   rdef->refc     = 1;
   
   SID_set(rdef, RDEF_SID);

   assert(rdef_is_valid(rdef));
   
   return rdef;
}

void rdef_free(RDef* rdef)
{
   assert(rdef_is_valid(rdef));

   rdef->refc--;
   
   if (rdef->refc == 0)
   {
      SID_del(rdef);

      free(rdef);
   }
}

bool rdef_is_valid(RDef const* rdef)
{
   return ((rdef != NULL)
      && SID_ok(rdef, RDEF_SID)
      && (rdef->filename != NULL)
      && (rdef->pattern  != NULL)
      && (rdef->comment  != NULL));
}

RDef* rdef_copy(RDef const* source)
{
   CLANG_WARN_OFF(-Wcast-qual)

   RDef* rdef = (RDef*)source;

   CLANG_WARN_ON
      
   assert(rdef_is_valid(rdef));

   rdef->refc++;

   return rdef;
}

void rdef_set_param(RDef* rdef, RPar const* rpar)
{
   assert(rdef_is_valid(rdef));
   assert(rpar_is_valid(rpar));

   switch(rpar->type)
   {
   case RPAR_SKIP :
      rdef->skip = rpar->val.i;
      break;
   case RPAR_USE :
      rdef->use  = rpar->val.i;
      break;
   case RPAR_CMNT :
      rdef->comment = rpar->val.s;
      break;
   case RPAR_MTCH :
      rdef->match = rpar->val.s;
      break;
   case RPAR_ERR :
   default :
      abort();
   }
}

char const* rdef_get_filename(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->filename;
}

char const* rdef_get_pattern(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->pattern;
}

char const* rdef_get_comment(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->comment;
}

char const* rdef_get_match(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->match;
}

int rdef_get_use(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->use;
}

int rdef_get_skip(RDef const* rdef)
{
   assert(rdef_is_valid(rdef));
   
   return rdef->skip;
}

/* ----------------------------------------------------------------------------
 * Read Parameter
 * ----------------------------------------------------------------------------
 */
RPar* rpar_new_skip(int skip)
{
   RPar* rpar = calloc(1, sizeof(*rpar));

   assert(rpar != NULL);

   rpar->type  = RPAR_SKIP;
   rpar->val.i = skip;
   
   SID_set(rpar, RPAR_SID);

   assert(rpar_is_valid(rpar));

   return rpar;
}

RPar* rpar_new_use(int use)
{
   RPar* rpar = calloc(1, sizeof(*rpar));

   assert(rpar != NULL);
   
   rpar->type  = RPAR_USE;
   rpar->val.i = use;

   SID_set(rpar, RPAR_SID);

   assert(rpar_is_valid(rpar));

   return rpar;
}

RPar* rpar_new_comment(char const* comment)
{
   RPar* rpar = calloc(1, sizeof(*rpar));

   assert(rpar    != NULL);
   assert(comment != NULL);
   
   rpar->type  = RPAR_CMNT;
   rpar->val.s = comment;
   
   SID_set(rpar, RPAR_SID);

   assert(rpar_is_valid(rpar));

   return rpar;
}

RPar* rpar_new_match(char const* match)
{
   RPar* rpar = calloc(1, sizeof(*rpar));

   assert(rpar  != NULL);
   assert(match != NULL);
   
   rpar->type  = RPAR_MTCH;
   rpar->val.s = match;
   
   SID_set(rpar, RPAR_SID);

   assert(rpar_is_valid(rpar));

   return rpar;
}

void rpar_free(RPar* rpar)
{
   assert(rpar_is_valid(rpar));

   SID_del(rpar);

   free(rpar);
}

bool rpar_is_valid(RPar const* rpar)
{
   return ((rpar != NULL) && SID_ok(rpar, RPAR_SID)
      && (rpar->type != RPAR_ERR));
}

RPar* rpar_copy(RPar const* rpar)
{
   RPar* rpnew = calloc(1, sizeof(*rpar));

   assert(rpar  != NULL);
   assert(rpnew != NULL);
   
   rpnew->type = rpar->type;
   rpnew->val  = rpar->val;
   
   SID_set(rpnew, RPAR_SID);

   assert(rpar_is_valid(rpnew));

   return rpnew;
}

