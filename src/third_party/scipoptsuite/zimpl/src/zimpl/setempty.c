/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: setempty.c                                                    */
/*   Name....: Set Empty Functions                                           */
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

#ifdef _MSC_VER
#pragma warning (disable: 4100) /* unreferenced formal parameter */
#endif

#define SET_EMPTY_SID          0x5345544f
#define SET_EMPTY_ITER_SID     0x53454f49

/* ------------------------------------------------------------------------- 
 * --- valid                 
 * -------------------------------------------------------------------------
 */
is_PURE
static bool set_empty_is_valid(Set const* set)
{
   return set != NULL
      && SID_ok2(set->empty, SET_EMPTY_SID)
      && set->head.refc > 0
      && set->head.members == 0;
}

static bool set_empty_iter_is_valid(SetIter const* iter)
{
   return iter != NULL && SID_ok2(iter->empty, SET_EMPTY_ITER_SID);
}

/* ------------------------------------------------------------------------- 
 * --- set_new                 
 * -------------------------------------------------------------------------
 */
Set* set_empty_new(int dim)
{
   Set* set;

   set = calloc(1, sizeof(*set));

   assert(set != NULL);

   set->head.refc    = 1;
   set->head.dim     = dim;
   set->head.members = 0;
   set->head.type    = SET_EMPTY;

   SID_set2(set->empty, SET_EMPTY_SID);

   assert(set_empty_is_valid(set));
   
   return set;
}

/* ------------------------------------------------------------------------- 
 * --- copy
 * -------------------------------------------------------------------------
 */
expects_NONNULL returns_NONNULL 
static Set* set_empty_copy(Set const* source)
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
static void set_empty_free(Set* set)
{
   assert(set_empty_is_valid(set));

   set->head.refc--;

   if (set->head.refc == 0)
   {
      SID_del2(set->empty);

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
static SetIterIdx set_empty_lookup_idx(Set const* set, Tuple const* tuple, int offset)
{
   assert(set_empty_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);

   return -1;
}

/* ------------------------------------------------------------------------- 
 * --- get_tuple                 
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL is_NORETURN
static void set_empty_get_tuple(
   Set const* set,
   SetIterIdx idx,
   Tuple*     tuple,
   int        offset)
{
   assert(set_empty_is_valid(set));
   assert(idx >= 0);
   assert(idx <= set->head.members);
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset + set->head.dim <= tuple_get_dim(tuple));

   fprintf(stderr, "internal_error: set_empty_get_tuple called\n");
   abort();
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
   
   assert(set_empty_is_valid(set));
   assert(pattern == NULL || tuple_is_valid(pattern));
   assert(offset      == 0);

   iter = calloc(1, sizeof(*iter));

   assert(iter != NULL);
   
   SID_set2(iter->empty, SET_EMPTY_ITER_SID);

   assert(set_empty_iter_is_valid(iter));

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
   SetIter*       iter,
   Set const*     set,
   is_UNUSED Tuple*  tuple,
   is_UNUSED int     offset)
{
   assert(set_empty_iter_is_valid(iter));
   assert(set_empty_is_valid(set));

   return false;
}

/* ------------------------------------------------------------------------- 
 * --- iter_exit
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL 
static void iter_exit(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_empty_iter_is_valid(iter));

   SID_del2(iter->empty);
   
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
   assert(set_empty_iter_is_valid(iter));
}

/* ------------------------------------------------------------------------- 
 * --- vtab_init
 * -------------------------------------------------------------------------
 */ 
void set_empty_init(SetVTab* vtab)
{
   vtab[SET_EMPTY].set_copy       = set_empty_copy;
   vtab[SET_EMPTY].set_free       = set_empty_free;
   vtab[SET_EMPTY].set_lookup_idx = set_empty_lookup_idx;
   vtab[SET_EMPTY].set_get_tuple  = set_empty_get_tuple;
   vtab[SET_EMPTY].iter_init      = iter_init;
   vtab[SET_EMPTY].iter_next      = iter_next;
   vtab[SET_EMPTY].iter_exit      = iter_exit;
   vtab[SET_EMPTY].iter_reset     = iter_reset;
   vtab[SET_EMPTY].set_is_valid   = set_empty_is_valid;
}




