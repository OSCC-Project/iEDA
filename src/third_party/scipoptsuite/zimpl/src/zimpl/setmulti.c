/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: setmulti.c                                                    */
/*   Name....: Set Multi Functions                                           */
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

#include <stddef.h>
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
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/list.h"
#include "zimpl/hash.h"
#include "zimpl/stmt.h"
#include "zimpl/set.h"
#include "zimpl/entry.h"
#include "zimpl/set4.h"

#ifdef _MSC_VER
#pragma warning (disable: 4100) /* unreferenced formal parameter */
#endif

#define SET_MULTI_SID          0x5345544d
#define SET_MULTI_ITER_SID     0x53454d49

/* This is a bloody hack. But there seems to be no easy way to give
 * additional information to the compare routine needed for qsort().
 */
static Set const* cmp_set = NULL;
static int        cmp_dim = 0;  

/* ------------------------------------------------------------------------- 
 * --- valid                 
 * -------------------------------------------------------------------------
 */
is_PURE
static bool set_multi_is_valid(Set const* set)
{
   return set != NULL
      && SID_ok2(set->multi, SET_MULTI_SID)
      && set->head.refc    > 0
      && set->head.dim     > 1
      && set->multi.subset != NULL
      && set->multi.set    != NULL;
}

is_PURE
static bool set_multi_iter_is_valid(SetIter const* iter)
{
   return iter != NULL
      && SID_ok2(iter->multi, SET_MULTI_ITER_SID)
      && iter->multi.members >= 0
      && iter->multi.now     >= 0
      && (  (iter->multi.members == 0 && iter->multi.dim == 0 && iter->multi.subset == NULL)
         || (iter->multi.members >= 0 && iter->multi.dim >  0 && iter->multi.subset != NULL));
}

/* ------------------------------------------------------------------------- 
 * --- internal                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL is_PURE
static int subset_cmp(void const* a, void const* b)
{
   SetIterIdx const* aa = (SetIterIdx const*)a;
   SetIterIdx const* bb = (SetIterIdx const*)b;

   for(int i = 0; i < cmp_dim; i++)
   {
      SetIterIdx d = aa[i] - bb[i];

      if (d < 0)
         return -1;
      if (d > 0)
         return 1;
   }
   return 0;
}

expects_NONNULL is_PURE
static int order_cmp(void const* a, void const* b)
{
   SetIterIdx const* aa = (SetIterIdx const*)a;
   SetIterIdx const* bb = (SetIterIdx const*)b;

   assert(cmp_set != NULL);
   assert(cmp_dim >= 0);
   assert(cmp_dim <  cmp_set->head.dim);
   
   SetIterIdx ai = *aa * cmp_set->head.dim + cmp_dim;
   SetIterIdx bi = *bb * cmp_set->head.dim + cmp_dim;

   SetIterIdx d = cmp_set->multi.subset[ai] - cmp_set->multi.subset[bi];

   if (d < 0)
      return -1;
   if (d > 0)
      return 1;

   return 0;
}

/* ------------------------------------------------------------------------- 
 * --- new                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL returns_NONNULL 
Set* set_multi_new_from_list(List const* list, SetCheckType check)
{
   assert(list_is_valid(list));

   ListElem*    le            = NULL;
   Hash*        hash          = NULL;     
   bool         is_entrylist  = list_is_entrylist(list); 
   int          n             = list_get_elems(list);
   Tuple const* tuple         = is_entrylist
      ? entry_get_tuple(list_get_entry(list, &le))
      : list_get_tuple(list, &le);
   int          dim           = tuple_get_dim(tuple);

   assert(n   > 0);
   assert(dim > 1);
   
   Set* set = calloc(1, sizeof(*set));
   
   assert(set != NULL);

   set->head.refc    = 1;
   set->head.dim     = dim;
   set->head.members = 0;
   set->head.type    = SET_MULTI;
   set->multi.set    = calloc((size_t)dim, sizeof(*set->multi.set));
   set->multi.subset = calloc((size_t)(n * dim), sizeof(*set->multi.subset));
   set->multi.order  = calloc((size_t)dim, sizeof(*set->multi.order));
      
   assert(set->multi.set    != NULL);
   assert(set->multi.subset != NULL);
   assert(set->multi.order  != NULL);
   
   for(int k = 0; k < dim; k++)
      set->multi.set[k] = set_list_new(n, SET_DEFAULT);

   if (check != SET_CHECK_NONE)
      hash = hash_new(HASH_TUPLE, n);
   
   le = NULL;
   
   for(int i = 0; i < n; i++)
   {
      tuple = is_entrylist
         ? entry_get_tuple(list_get_entry(list, &le))
         : list_get_tuple(list, &le);

      assert(tuple != NULL);
      assert(hash != NULL || check == SET_CHECK_NONE);

      if (hash != NULL && hash_has_tuple(hash, tuple))
      {
         if (check == SET_CHECK_WARN)
         {
            if (stmt_trigger_warning(164))
            {
               fprintf(stderr, "--- Warning 164: Duplicate element ");
               tuple_print(stderr, tuple);
               fprintf(stderr, " for set rejected\n");
            }
         }
      }
      else
      {
         if (hash != NULL)
            hash_add_tuple(hash, tuple);

         for(int k = 0; k < dim; k++)
            set->multi.subset[set->head.members * dim + k] =
               set_list_add_elem(set->multi.set[k],
                  tuple_get_elem(tuple, k), SET_CHECK_QUIET);

         set->head.members++;
      }
   }
   if (hash != NULL)
      hash_free(hash);
   
   /* Bloody hack!
    */
   cmp_set = set;
   cmp_dim = dim;

   /* Sort subset
    */
   qsort(set->multi.subset, (size_t)set->head.members,
      (size_t)dim * sizeof(*set->multi.subset), subset_cmp);
   
   /* This could be done also later On-Demand
    */
   for(int k = 0; k < dim; k++)
   {
      set->multi.order[k] = calloc((size_t)set->head.members, sizeof(**set->multi.order));
      
      assert(set->multi.order[k] != NULL);

      for(SetIterIdx i = 0; i < set->head.members; i++)
         set->multi.order[k][i] = i;

      /* No need to sort order[0] because subset ist already sorted.
       */
      if (k > 0)
      {
         cmp_dim = k;
      
         qsort(set->multi.order[k], (size_t)set->head.members,
            sizeof(**set->multi.order), order_cmp);
      }
#ifndef NDEBUG
      /* Make sure order is sorted
       */
      for(SetIterIdx i = 0; i < set->head.members - 1; i++)
      {
         SetIterIdx a = set->multi.order[k][i]     * set->head.dim + k;
         SetIterIdx b = set->multi.order[k][i + 1] * set->head.dim + k;
         
         assert(set->multi.subset[a] <= set->multi.subset[b]);
      }
#endif /* !NDEBUG */
   }
   SID_set2(set->multi, SET_MULTI_SID);

   assert(set_multi_is_valid(set));

#ifndef NDEBUG
   le = NULL;
   
   for(int i = 0; i < n; i++)
   {
      tuple = is_entrylist
         ? entry_get_tuple(list_get_entry(list, &le))
         : list_get_tuple(list, &le);

      assert(set_lookup(set, tuple));
   }
#endif
   
   return set;
}

/* ------------------------------------------------------------------------- 
 * --- copy
 * -------------------------------------------------------------------------
 */
expects_NONNULL returns_NONNULL 
static Set* set_multi_copy(Set const* source)
{
   CLANG_WARN_OFF(-Wcast-qual)
      
   Set* set = (Set*)source;

   CLANG_WARN_ON
   
   assert(set_multi_is_valid(source));

   set->head.refc++;

   for(int i = 0; i < set->head.dim; i++)
      (void)set_copy(set->multi.set[i]);
   
   return set;
}

/* ------------------------------------------------------------------------- 
 * --- free                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL 
static void set_multi_free(Set* set)
{
   int i;
   
   assert(set_multi_is_valid(set));

   for(i = 0; i < set->head.dim; i++)
      set_free(set->multi.set[i]);

   set->head.refc--;

   if (set->head.refc == 0)
   {
      SID_del2(set->multi);

      for(i = 0; i < set->head.dim; i++)
         free(set->multi.order[i]);

      free(set->multi.order);
      free(set->multi.set);
      free(set->multi.subset);
      free(set);
   }
}

/* ------------------------------------------------------------------------- 
 * --- lookup                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL is_PURE
static int subset_idx_cmp(void const* a, void const* b)
{
   SetIterIdx const* key    = (SetIterIdx const*)a;
   SetIterIdx const* subset = (SetIterIdx const*)b;
   int               i;
   
   assert(key    != NULL);
   assert(subset != NULL);
   
   assert(cmp_dim > 0);

   for(i = 0; i < cmp_dim; i++)
   {
      SetIterIdx d = key[i] - subset[i];

      if (d < 0)
         return -1;
      if (d > 0)
         return 1;
   }
   return 0;
}

expects_NONNULL is_PURE
static int order_idx_cmp(void const* a, void const* b)
{
   SetIterIdx const* key   = (SetIterIdx const*)a;
   SetIterIdx const* order = (SetIterIdx const*)b;

   assert(key     != NULL);
   assert(order   != NULL);
   assert(cmp_set != NULL);

   assert(cmp_dim >= 0);
   assert(cmp_dim <  cmp_set->head.dim);
   assert(*order  >= 0);
   assert(*order  <  cmp_set->head.members);

   SetIterIdx d = *key - cmp_set->multi.subset[*order * cmp_set->head.dim + cmp_dim];

   if (d < 0)
      return -1;
   if (d > 0)
      return 1;
   return 0;
}

/* return the index of the element, -1 if not found
 */
expects_NONNULL 
static SetIterIdx set_multi_lookup_idx(Set const* set, Tuple const* tuple, int offset)
{
   SetIterIdx* idx;
   ptrdiff_t   result;
   
   assert(set_multi_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset <  tuple_get_dim(tuple));

   idx = malloc((size_t)set->head.dim * sizeof(*idx));

   assert(idx != NULL);

   for(int i = 0; i < set->head.dim; i++)
   {
      idx[i] = set_lookup_idx(set->multi.set[i], tuple, offset + i);

      if (idx[i] < 0)
      {
         free(idx);
         return -1;
      }
   }
   cmp_dim = set->head.dim;
   
   result = (ptrdiff_t)bsearch(idx, set->multi.subset, (size_t)set->head.members,
      (size_t)set->head.dim * sizeof(*set->multi.subset), subset_idx_cmp);
   
   free(idx);

   if (result == 0)
      return -1;

   assert((result - (ptrdiff_t)set->multi.subset)
      % (ptrdiff_t)((size_t)set->head.dim * sizeof(*set->multi.subset)) == 0);

   SetIterIdx k = (SetIterIdx)(result - (ptrdiff_t)set->multi.subset)
      / (ptrdiff_t)((size_t)set->head.dim * sizeof(*set->multi.subset));

   assert(k >= 0);
   assert(k <  set->head.members);

   return k;
}

/* ------------------------------------------------------------------------- 
 * --- get_tuple                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL 
static void set_multi_get_tuple(
   Set const* set,
   SetIterIdx idx,
   Tuple*     tuple,
   int        offset)
{
   assert(set_multi_is_valid(set));
   assert(idx >= 0);
   assert(idx <= set->head.members);
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset + set->head.dim <= tuple_get_dim(tuple));

   for(int i = 0; i < set->head.dim; i++)
   {
      tuple_set_elem(tuple, offset + i,
         elem_copy(set_list_get_elem(set->multi.set[i],
            set->multi.subset[idx * set->head.dim + i])));
   }
}

/* ------------------------------------------------------------------------- 
 * --- iter_init                 
 * -------------------------------------------------------------------------
 */
expects_NONNULL1 returns_NONNULL 
static SetIter* set_multi_iter_init(
   Set const*   set,
   Tuple const* pattern,
   int          offset)
{
   int          m;
   int          first;
   int          last;
   ptrdiff_t    result;
   int          fixed_idx = -1;
   
   assert(set_multi_is_valid(set));
   assert(pattern == NULL || tuple_is_valid(pattern));
   assert(offset      >= 0);
   assert(pattern == NULL || offset < tuple_get_dim(pattern));

   SetIter*    iter = calloc(1, sizeof(*iter));
   SetIterIdx* idx  = malloc((size_t)set->head.dim * sizeof(*idx));
   int         i;
   
   for(i = 0; i < set->head.dim; i++)
   {
      if (pattern == NULL
         || elem_get_type(tuple_get_elem(pattern, offset + i)) == ELEM_NAME)
         idx[i] = -1;
      else
      {
         idx[i] = set_lookup_idx(set->multi.set[i], pattern, offset + i);
         
         if (idx[i] < 0)
            break;

         fixed_idx = i;
      }
   }
   /* Impossible pattern ?
    */
   if (i < set->head.dim)
   {
      iter->multi.members = 0;
      iter->multi.now     = 0;
   }
   else
   {
      iter->multi.dim = set->head.dim;
   
      iter->multi.subset = calloc((size_t)set->head.members, 
         (size_t)set->head.dim * sizeof(*iter->multi.subset));

      assert(iter->multi.subset);

      /* Pattern matches all members ?
       */
      if (fixed_idx < 0)
      {
         iter->multi.members = set->head.members;
         iter->multi.now     = 0;

         memcpy(iter->multi.subset, set->multi.subset,
            (size_t)(iter->multi.members * iter->multi.dim) * sizeof(*iter->multi.subset));
      }
      else
      {
         assert(fixed_idx >= 0);

         cmp_set = set;
         cmp_dim = fixed_idx;
         
         result = (ptrdiff_t)bsearch(
            &idx[fixed_idx],
            set->multi.order[fixed_idx],
            (size_t)set->head.members,
            sizeof(**set->multi.order),
            order_idx_cmp);

#if 0
         if (result == 0)
         {
            for(int i = 0; i < set->head.members; i++)
               fprintf(stderr, "%d %d %d\n", i, set->multi.order[fixed_idx][i],
                  set->multi.subset[set->multi.order[fixed_idx][i] * set->head.dim + fixed_idx]);
         }
#endif
         assert(result != 0);

         int k = (int)(result - (ptrdiff_t)set->multi.order[fixed_idx])
            / (ptrdiff_t)sizeof(**set->multi.order);

         assert(k >= 0);
         assert(k <  set->head.members);

         assert(set->multi.order[fixed_idx][k] >= 0);
         assert(set->multi.order[fixed_idx][k] <  set->head.members);
         
         assert(idx[fixed_idx] == set->multi.subset[set->multi.order[fixed_idx][k] * set->head.dim + fixed_idx]);

#if 0
         fprintf(stderr, "@ fixe_idx: %d idx[]=%d k=%d j=%d\n",
            fixed_idx, idx[fixed_idx], k, j);
#endif     
         for(first = k; first >= 0; first--)
         {
            SetIterIdx j = set->multi.order[fixed_idx][first] * set->head.dim + fixed_idx;
            
            if (idx[fixed_idx] != set->multi.subset[j])
            {
               assert(idx[fixed_idx] > set->multi.subset[j]);
               break;
            }
         }
         for(last = k; last < set->head.members; last++)
         {
            SetIterIdx j = set->multi.order[fixed_idx][last] * set->head.dim + fixed_idx;
            
            if (idx[fixed_idx] != set->multi.subset[j])
            {
               assert(idx[fixed_idx] < set->multi.subset[j]);
               break;
            }
         }
         assert(first + 1 < last);
         
         m = 0;

         for(k = first + 1; k < last; k++)
         {
            SetIterIdx j = set->multi.order[fixed_idx][k];
            
            assert(idx[fixed_idx] == set->multi.subset[j * set->head.dim + fixed_idx]);

            for(i = 0; i < set->head.dim; i++)
            {
               /* can entry match ?
                */
               if (idx[i] >= 0 && idx[i] != set->multi.subset[j * set->head.dim + i])
                  break;

               iter->multi.subset[m * set->head.dim + i] =
                  set->multi.subset[j * set->head.dim + i];
            }
            if (i == set->head.dim)
               m++;
         }
         iter->multi.members = m;
         iter->multi.now     = 0;
      }
   }
#if 0
   fprintf(stderr, "set_multi_iter_init dim=%d members=%d\n",
      set->head.dim, iter->multi.members);

   for(i = 0; i < set->head.dim; i++)
      fprintf(stderr, "idx[%d]=%d\n", i, idx[i]);
#endif
   free(idx);
   
   SID_set2(iter->multi, SET_MULTI_ITER_SID);

   assert(set_multi_iter_is_valid(iter));

   return iter;
}

/* ------------------------------------------------------------------------- 
 * --- iter_next
 * -------------------------------------------------------------------------
 */
/* false means, there is no further element
 */
expects_NONNULL 
static bool set_multi_iter_next(
   SetIter*   iter,
   Set const* set,
   Tuple*     tuple,
   int        offset)
{
   assert(set_multi_iter_is_valid(iter));
   assert(set_multi_is_valid(set));
   assert(tuple_is_valid(tuple));
   assert(offset >= 0);
   assert(offset + set->head.dim <= tuple_get_dim(tuple));

   if (iter->multi.now >= iter->multi.members)
      return false;

   for(int i = 0; i < iter->multi.dim; i++)
      tuple_set_elem(tuple, offset + i,
         elem_copy(set_list_get_elem(set->multi.set[i],
            iter->multi.subset[iter->multi.now * iter->multi.dim + i])));

   iter->multi.now++;
   
   return true;
}

/* ------------------------------------------------------------------------- 
 * --- iter_exit
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL
static void set_multi_iter_exit(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_multi_iter_is_valid(iter));

   SID_del2(iter->multi);
   
   if (iter->multi.subset != NULL)
      free(iter->multi.subset);

   free(iter);
}

/* ------------------------------------------------------------------------- 
 * --- iter_reset
 * -------------------------------------------------------------------------
 */
/*ARGSUSED*/
expects_NONNULL 
static void set_multi_iter_reset(SetIter* iter, is_UNUSED Set const* set)
{
   assert(set_multi_iter_is_valid(iter));
   
   iter->multi.now = 0;
}

/* ------------------------------------------------------------------------- 
 * --- vtab_init
 * -------------------------------------------------------------------------
 */
void set_multi_init(SetVTab* vtab)
{
   vtab[SET_MULTI].set_copy       = set_multi_copy;
   vtab[SET_MULTI].set_free       = set_multi_free;
   vtab[SET_MULTI].set_lookup_idx = set_multi_lookup_idx;
   vtab[SET_MULTI].set_get_tuple  = set_multi_get_tuple;
   vtab[SET_MULTI].iter_init      = set_multi_iter_init;
   vtab[SET_MULTI].iter_next      = set_multi_iter_next;
   vtab[SET_MULTI].iter_exit      = set_multi_iter_exit;
   vtab[SET_MULTI].iter_reset     = set_multi_iter_reset;
   vtab[SET_MULTI].set_is_valid   = set_multi_is_valid;
}







