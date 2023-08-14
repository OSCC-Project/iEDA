/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: bound.c                                                       */
/*   Name....: Bound value                                                   */
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
#include "zimpl/numb.h"
#include "zimpl/bound.h"

#define BOUND_SID     0x426F756E

struct bound
{
   SID
   BoundType type;
   Numb*     value;
};

Bound* bound_new(BoundType type, Numb const* value)
{
   Bound* bound = calloc(1, sizeof(*bound));

   assert(bound != NULL);

   bound->type = type;

   if (bound->type == BOUND_VALUE)
   {
      assert(value != NULL);

      bound->value = numb_copy(value);
   }
   SID_set(bound, BOUND_SID);

   assert(bound_is_valid(bound));
   
   return bound;
}

void bound_free(Bound* bound)
{
   assert(bound_is_valid(bound));

   if (bound->type == BOUND_VALUE)
      numb_free(bound->value);
   
   SID_del(bound);

   free(bound);
}

bool bound_is_valid(Bound const* bound)
{
   if (bound == NULL || !SID_ok(bound, BOUND_SID)
      || (bound->type == BOUND_VALUE && bound->value == NULL)
      || (bound->type != BOUND_VALUE && bound->value != NULL))
      return false;

   mem_check(bound);

   return true;
}

Bound* bound_copy(Bound const* source)
{
   assert(bound_is_valid(source));

   return bound_new(source->type, source->value);
}

BoundType bound_get_type(Bound const* bound)
{
   assert(bound_is_valid(bound));

   return bound->type;
}

Numb const* bound_get_value(Bound const* bound)
{
   assert(bound_is_valid(bound));
   assert(bound->type == BOUND_VALUE);
   
   return bound->value;
}

void bound_print(FILE* fp, Bound const* bound)
{
   switch(bound->type)
   {
   case BOUND_INFTY :
      fprintf(fp, "oo");
      break;
   case BOUND_MINUS_INFTY :
      fprintf(fp, "-oo");
      break;
   case BOUND_VALUE :
      numb_print(fp, bound->value);
      break;
   default :
      abort();
   }
}
