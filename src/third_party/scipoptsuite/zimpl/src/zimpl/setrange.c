/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: setrange.c                                                    */
/*   Name....: Set Range Functions                                           */
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
#include "zimpl/stmt.h"
#include "zimpl/set.h"
#include "zimpl/set4.h"

#ifdef _MSC_VER
#pragma warning (disable: 4100) /* unreferenced formal parameter */
#endif

#define SET_RANGE_SID          0x53455452
#define SET_RANGE_ITER_SID     0x53455249

/* ------------------------------------------------------------------------- 
 * --- valid                 
 * -------------------------------------------------------------------------
 */
is_PURE
static bool set_range_is_valid(Set const* set)
{
   return set != NULL
      && SID_ok2(set->range, SET_RANGE_SID)
      && set->head.refc > 0
      && set->head.dim == 1;
}

is_PURE
static bool set_range_iter_is_valid(SetIter const* iter)
{
   return iter != NULL && SID_ok2(iter->range, SET_RANGE_ITER_SID)
      && iter->range.first >= 0
      && iter->range.last  >= 0
      && iter->range.now   >= iter->range.first;
}

/* ------------------------------------------------------------------------- 
 * --- set_new                 
 * -------------------------------------------------------------------------
 */
returns_NONNULL 
Set* set_range_new(int begin, int end, int step)
{
   Set* set;

   set = calloc(1, sizeof(*set));

   assert(set != NULL);

   set->head.refc    = 1;
   set->head.dim     = 1;
   set->head.members = 1 + (end - begin) / step;
   set->head.type    = SET_RANGE;

   set->range.begin  = begin;
   set->range.end    = end;
   set->range.step   = step;

   SID_set2(set->range, SET_RANGE_SID);

   assert(set_range_is_valid(set));
   
   return set;
}

/* ------------------------------------------------------------------------- 
 * --- copy
 * -------------------------------------------------------------------------
 */
expects_NONNULL returns_NONNULL 
static Set* set_range_copy(Set const* source)
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
static void set_range_free(Set* set)
{
   assert(set_range_is_valid(set));

   set->head.refc--;

   if (set->head.refc == 0)
   {
      SID_del2(set->range);

      free(set);
   }
}

/* ------------------------------------------------------------------------- 
 * --- lookup                 
 * -------------------------------------------------------------------------
 */
/* Return index number of element. -1 if not present
 */
is_CONST
static long long idx_to_val(SetIterIdx begin, SetIterIdx step, SetIterIdx idx)
{
#if 0
   fprintf(stderr, "idx_to_val: %lld %lld %lld = %lld\n",
      begin, step, idx, begin + idx * step);
#endif
   return begin + idx * step;
}

is_CONST
static SetIterIdx val_to_idx(SetIterIdx begin, SetIterIdx step, SetIterIdx val)
{
#if 0
   fprintf(stderr, "val_to_idx: %lld %lld %lld = %lld\n",
      begin, step, val, (val - begin) / step);
#endif
   return (val - begin) / step;
}

expects_NONNULL 
static SetIterIdx set_range_lookup_idx(Set const* set, Tuple const* tuple, int offset)
{
   Elem const* elem;
   Numb const* numb;
   int         val;
   
   assert(set_range_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset <  tuple_get_dim(tuple));
   
   elem = tuple_get_elem(tuple, offset);

   /* If this is true, we asked a number set for a string.
    */
   if (elem_get_type(elem) != ELEM_NUMB)
      return -1;

   numb = elem_get_numb(elem);

   assert(numb_is_int(numb));

   val = numb_toint(numb);

   if (set->range.step > 0)
   {
      if (  val < set->range.begin 
         || val > set->range.end
         || ((val - set->range.begin) % set->range.step) != 0)
         return -1;
   }
   else
   {
      assert(set->range.step < 0);
      
      if (  val > set->range.begin 
         || val < set->range.end
         || ((set->range.begin - val) % set->range.step) != 0)
         return -1;
   }
   return val_to_idx(set->range.begin, set->range.step, val);
}

/* ------------------------------------------------------------------------- 
 * --- get_tuple                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL 
static void set_range_get_tuple(
   Set const* set,
   SetIterIdx idx,
   Tuple*     tuple,
   int        offset)
{
   assert(set_range_is_valid(set));
   assert(idx >= 0);
   assert(idx <= set->head.members);
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset <  tuple_get_dim(tuple));

   Numb* numb = numb_new_longlong(idx_to_val(set->range.begin, set->range.step, idx));

   tuple_set_elem(tuple, offset, elem_new_numb(numb));

   numb_free(numb);
}

/* ------------------------------------------------------------------------- 
 * --- iter_init                 
 * -------------------------------------------------------------------------
 */
/* Initialise Iterator. Write into iter
 */
expects_NONNULL1 returns_NONNULL 
static SetIter* set_range_iter_init(
   Set const*   set,
   Tuple const* pattern,
   int          offset)
{
   Elem const*  elem;
   SetIter*     iter;
   
   assert(set_range_is_valid(set));
   assert(pattern == NULL || tuple_is_valid(pattern));
   assert(offset      >= 0);
   assert(pattern == NULL || offset <  tuple_get_dim(pattern));

   iter = calloc(1, sizeof(*iter));

   assert(iter != NULL);

   if (pattern == NULL)
   {
      iter->range.first = 0;
      iter->range.last  = val_to_idx(set->range.begin, set->range.step, set->range.end);
   }
   else
   {
      elem = tuple_get_elem(pattern, offset);

      switch(elem_get_type(elem))
      {
      case ELEM_NAME :
         iter->range.first = 0;
         iter->range.last  = val_to_idx(set->range.begin, set->range.step, set->range.end);
         break;
      case ELEM_NUMB :
         iter->range.first = set_range_lookup_idx(set, pattern, offset);

         if (iter->range.first >= 0)
            iter->range.last = iter->range.first;
         else
         {
            iter->range.first = 1;
            iter->range.last  = 0;
         }
         break;
      case ELEM_STRG :
         /* This should not happen. Probably a set with mixed
          * numbers and string was generated.
          */
      default :
         abort();
      }
   }
   iter->range.now = iter->range.first;

   SID_set2(iter->range, SET_RANGE_ITER_SID);

   assert(set_range_iter_is_valid(iter));

   return iter;
}

/* ------------------------------------------------------------------------- 
 * --- iter_next
 * -------------------------------------------------------------------------
 */
/* false means, there is no further element
 */
expects_NONNULL 
static bool set_range_iter_next(
   SetIter*   iter,
   Set const* set,
   Tuple*     tuple,
   int        offset)
{
   assert(set_range_iter_is_valid(iter));
   assert(set_range_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset <  tuple_get_dim(tuple));

   if (iter->range.now > iter->range.last)
      return false;

   Numb* numb = numb_new_longlong(idx_to_val(set->range.begin, set->range.step, iter->range.now));

   tuple_set_elem(tuple, offset, elem_new_numb(numb));

   numb_free(numb);

   iter->range.now++;

   return true;
}

/* ------------------------------------------------------------------------- 
 * --- iter_exit
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL 
static void set_range_iter_exit(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_range_iter_is_valid(iter));

   SID_del2(iter->range);
   
   free(iter);
}

/* ------------------------------------------------------------------------- 
 * --- iter_reset
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL 
static void set_range_iter_reset(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_range_iter_is_valid(iter));
   
   iter->range.now = iter->range.first;
}

/* ------------------------------------------------------------------------- 
 * --- vtab_init
 * -------------------------------------------------------------------------
 */
void set_range_init(SetVTab* vtab)
{
   vtab[SET_RANGE].set_copy       = set_range_copy;
   vtab[SET_RANGE].set_free       = set_range_free;
   vtab[SET_RANGE].set_lookup_idx = set_range_lookup_idx;
   vtab[SET_RANGE].set_get_tuple  = set_range_get_tuple;
   vtab[SET_RANGE].iter_init      = set_range_iter_init;
   vtab[SET_RANGE].iter_next      = set_range_iter_next;
   vtab[SET_RANGE].iter_exit      = set_range_iter_exit;
   vtab[SET_RANGE].iter_reset     = set_range_iter_reset;
   vtab[SET_RANGE].set_is_valid   = set_range_is_valid;
}


