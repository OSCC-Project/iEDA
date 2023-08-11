/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: setpseudo.c                                                    */
/*   Name....: Set Pseudo Functions                                           */
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

#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/hash.h"
#include "zimpl/set.h"
#include "zimpl/set4.h"

#define SET_PSEUDO_SID          0x53455452
#define SET_PSEUDO_ITER_SID     0x53455249

/* ------------------------------------------------------------------------- 
 * --- valid                 
 * -------------------------------------------------------------------------
 */
is_PURE
static bool set_pseudo_is_valid(Set const* set)
{
   return set != NULL
      && SID_ok2(set->pseudo, SET_PSEUDO_SID)
      && set->head.refc > 0
      && set->head.dim     == 0
      && set->head.members == 1;
}

static bool set_pseudo_iter_is_valid(SetIter const* iter)
{
   return iter != NULL && SID_ok2(iter->pseudo, SET_PSEUDO_ITER_SID);
}

/* ------------------------------------------------------------------------- 
 * --- set_new                 
 * -------------------------------------------------------------------------
 */
Set* set_pseudo_new()
{
   Set* set;

   set = calloc(1, sizeof(*set));

   assert(set != NULL);

   set->head.refc    = 1;
   set->head.dim     = 0;
   set->head.members = 1;
   set->head.type    = SET_PSEUDO;

   SID_set2(set->pseudo, SET_PSEUDO_SID);

   assert(set_pseudo_is_valid(set));
   
   return set;
}

/* ------------------------------------------------------------------------- 
 * --- copy
 * -------------------------------------------------------------------------
 */
expects_NONNULL returns_NONNULL 
static Set* set_pseudo_copy(Set const* source)
{
   CLANG_WARN_OFF(-Wcast-qual)

   Set* set = (Set*)source;

   CLANG_WARN_ON
      
   set->head.refc++;

   return set;
}

/* ------------------------------------------------------------------------- 
 * --- set_free                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL
static void set_pseudo_free(Set* set)
{
   assert(set_pseudo_is_valid(set));

   set->head.refc--;

   if (set->head.refc == 0)
   {
      SID_del2(set->pseudo);

      free(set);
   }
}

/* ------------------------------------------------------------------------- 
 * --- lookup                 
 * -------------------------------------------------------------------------
 */
/* Return index number of element. -1 if not present
 */
/*ARGSUSED*/
expects_NONNULL
static SetIterIdx set_pseudo_lookup_idx(Set const* set, Tuple const* tuple, int offset)
{
   assert(set_pseudo_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset == 0);
   assert(tuple_get_dim(tuple) == 0);

   return 0;
}

/* ------------------------------------------------------------------------- 
 * --- get_tuple                 
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL
static void set_pseudo_get_tuple(
   Set const* set,
   SetIterIdx idx,
   Tuple*     tuple,
   int        offset)
{
   assert(set_pseudo_is_valid(set));
   assert(idx == 0);
   assert(tuple_is_valid(tuple));
   assert(offset == 0);
   assert(tuple_get_dim(tuple) == 0);
}

/* ------------------------------------------------------------------------- 
 * --- iter_init                 
 * -------------------------------------------------------------------------
 */
/* Initialise Iterator. Write into iter
 */
/*ARGSUSED*/
expects_NONNULL1 returns_NONNULL
static SetIter* iter_init(
   Set const*   set,
   Tuple const* pattern,
   int          offset)
{
   SetIter*        iter;
   
   assert(set_pseudo_is_valid(set));
   assert(pattern == NULL || tuple_is_valid(pattern));
   assert(pattern == NULL || tuple_get_dim(pattern) == 0);
   assert(offset                 == 0);

   iter = calloc(1, sizeof(*iter));

   assert(iter != NULL);

   iter->pseudo.first = true;
   
   SID_set2(iter->pseudo, SET_PSEUDO_ITER_SID);

   assert(set_pseudo_iter_is_valid(iter));

   return iter;
}

/* ------------------------------------------------------------------------- 
 * --- iter_next
 * -------------------------------------------------------------------------
 */
/* false means, there is no further element
 */
/*ARGSUSED*/
expects_NONNULL
static bool iter_next(
   SetIter*             iter,
   is_UNUSED Set const* set,
   is_UNUSED Tuple*     tuple,
   is_UNUSED int        offset)
{
   assert(set_pseudo_iter_is_valid(iter));

   if (!iter->pseudo.first)
      return false;

   iter->pseudo.first = false;
   
   return true;
}

/* ------------------------------------------------------------------------- 
 * --- iter_exit
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL
static void iter_exit(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_pseudo_iter_is_valid(iter));

   SID_del2(iter->pseudo);

   free(iter);
}

/* ------------------------------------------------------------------------- 
 * --- iter_reset
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL
static void iter_reset(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_pseudo_iter_is_valid(iter));

   iter->pseudo.first = true;
}

/* ------------------------------------------------------------------------- 
 * --- vtab_init
 * -------------------------------------------------------------------------
 */
void set_pseudo_init(SetVTab* vtab)
{
   vtab[SET_PSEUDO].set_copy       = set_pseudo_copy;
   vtab[SET_PSEUDO].set_free       = set_pseudo_free;
   vtab[SET_PSEUDO].set_lookup_idx = set_pseudo_lookup_idx;
   vtab[SET_PSEUDO].set_get_tuple  = set_pseudo_get_tuple;
   vtab[SET_PSEUDO].iter_init      = iter_init;
   vtab[SET_PSEUDO].iter_next      = iter_next;
   vtab[SET_PSEUDO].iter_exit      = iter_exit;
   vtab[SET_PSEUDO].iter_reset     = iter_reset;
   vtab[SET_PSEUDO].set_is_valid   = set_pseudo_is_valid;
}







