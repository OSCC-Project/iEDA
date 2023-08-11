/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: hash.c                                                        */
/*   Name....: Hash Functions                                                */
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

#include "zimpl/blkmem.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/mono.h"
#include "zimpl/hash.h"

#define HASH_SID      0x48617368

typedef struct hash_element HElem;
typedef struct set_elem_idx SetElemIdx;

struct set_elem_idx
{
   Elem const* elem;
   int         idx;
};

struct hash_element
{
   union
   {
      Tuple const* tuple;
      Entry const* entry;
      SetElemIdx   elem_idx;
      Numb const*  numb;
      Mono const*  mono;
   } value;
   HElem* next;
};

struct hash
{
   SID
   unsigned int size;
   int          elems;
   HashType     type;
   HElem**      bucket;
};

static void hash_statist(FILE* fp, Hash const* hash);

is_PURE
static bool hash_is_valid(Hash const* hash)
{
   return ((hash != NULL)
      && (hash->type == HASH_TUPLE || hash->type == HASH_ENTRY
       || hash->type == HASH_ELEM_IDX || hash->type == HASH_NUMB
       || hash->type == HASH_MONO)
      && SID_ok(hash, HASH_SID));
}

Hash* hash_new(HashType type, int size)
{
   static const unsigned int bucket_size[] =
   {
      53U, 103U, 503U, 1009U, 5003U, 10007U, 50021U, 100003U, 500009U, 1000003U, 
      5000011U, 10000019U, 50000017U, 0U
   };
   
   Hash* hash = calloc(1, sizeof(*hash));
   int   i;
   
   assert(hash != NULL);
   assert(size >= 0);

   /* This is a linear search, but if the number of elements is large,
    * it will hardly matter ;-)
    */
   for(i = 0; bucket_size[i] < (unsigned int)size && bucket_size[i + 1] != 0; i++)
      ;

   hash->size = bucket_size[i];

   assert(hash->size > 11);
   
   hash->elems  = 0;
   hash->type   = type;
   hash->bucket = calloc(hash->size, sizeof(*hash->bucket));

   assert(hash->bucket != NULL);

   SID_set(hash, HASH_SID);
   assert(hash_is_valid(hash));
   
   return hash;
}

void hash_free(Hash* hash)
{
   HElem*       he;
   HElem*       hq;
   unsigned int i;
      
   assert(hash_is_valid(hash));

   if (verbose >= VERB_CHATTER)
      hash_statist(stderr, hash);
   
   SID_del(hash);

   for(i = 0; i < hash->size; i++)
   {
      for(he = hash->bucket[i]; he != NULL; he = hq)
      {
         hq = he->next;
         blk_free(he, sizeof(*he));
      }
   }
   free(hash->bucket);
   free(hash);
}

void hash_add_tuple(Hash* hash, Tuple const* tuple)
{
   HElem*       he = blk_alloc(sizeof(*he));
   unsigned int hcode;

   assert(hash_is_valid(hash));
   assert(tuple_is_valid(tuple));
   assert(hash->type == HASH_TUPLE);
   assert(he != NULL);
   
   hcode               = tuple_hash(tuple) % hash->size;
   he->value.tuple     = tuple;
   he->next            = hash->bucket[hcode];
   hash->bucket[hcode] = he;
   hash->elems++;
}


void hash_add_entry(Hash* hash, Entry const* entry)
{
   HElem*       he = blk_alloc(sizeof(*he));
   Tuple const* tuple;
   unsigned int hcode;

   assert(hash_is_valid(hash));
   assert(entry_is_valid(entry));
   assert(hash->type == HASH_ENTRY);
   assert(he != NULL);
   
   tuple               = entry_get_tuple(entry);
   hcode               = tuple_hash(tuple) % hash->size;
   he->value.entry     = entry;
   he->next            = hash->bucket[hcode];
   hash->bucket[hcode] = he;
   hash->elems++;
}

void hash_add_numb(Hash* hash, Numb const* numb)
{
   HElem*       he = blk_alloc(sizeof(*he));
   unsigned int hcode;

   assert(hash_is_valid(hash));
   assert(numb_is_valid(numb));
   assert(hash->type == HASH_NUMB);
   assert(he != NULL);
   
   hcode               = numb_hash(numb) % hash->size;
   he->value.numb      = numb;
   he->next            = hash->bucket[hcode];
   hash->bucket[hcode] = he;
   hash->elems++;
}

void hash_add_mono(Hash* hash, Mono const* mono)
{
   HElem*       he = blk_alloc(sizeof(*he));
   unsigned int hcode;

   assert(hash_is_valid(hash));
   assert(mono_is_valid(mono));
   assert(hash->type == HASH_MONO);
   assert(he != NULL);
   
   hcode               = mono_hash(mono) % hash->size;
   he->value.mono      = mono;
   he->next            = hash->bucket[hcode];
   hash->bucket[hcode] = he;
   hash->elems++;
}

bool hash_has_tuple(Hash const* hash, Tuple const* tuple)
{
   unsigned int hcode = tuple_hash(tuple) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(tuple_is_valid(tuple));

   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (!tuple_cmp(he->value.tuple, tuple))
         break;

   return he != NULL;
}

bool hash_has_entry(Hash const* hash, Tuple const* tuple)
{
   unsigned int hcode = tuple_hash(tuple) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(tuple_is_valid(tuple));
   
   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (!entry_cmp(he->value.entry, tuple))
         break;

   return he != NULL;
}

bool hash_has_numb(Hash const* hash, Numb const* numb)
{
   unsigned int hcode = numb_hash(numb) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(numb_is_valid(numb));

   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (numb_equal(he->value.numb, numb))
         break;

   return he != NULL;
}

/* Liefert NULL wenn nicht gefunden.
 */
Entry const* hash_lookup_entry(Hash const* hash, Tuple const* tuple)
{
   unsigned int hcode = tuple_hash(tuple) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(tuple_is_valid(tuple));

   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (!entry_cmp(he->value.entry, tuple))
         break;

   if (he == NULL)
      return NULL;

   assert(he != NULL);

   assert(entry_is_valid(he->value.entry));

   return he->value.entry;
}

/* Liefert NULL wenn nicht gefunden.
 */
Mono const* hash_lookup_mono(Hash const* hash, Mono const* mono)
{
   unsigned int hcode = mono_hash(mono) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(mono_is_valid(mono));

   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (mono_equal(he->value.mono, mono))
         break;

   if (he == NULL)
      return NULL;

   assert(he != NULL);

   assert(mono_is_valid(he->value.mono));

   return he->value.mono;
}

void hash_add_elem_idx(Hash* hash, Elem const* elem, int idx)
{
   HElem*       he = blk_alloc(sizeof(*he));
   unsigned int hcode;

   assert(hash_is_valid(hash));
   assert(elem_is_valid(elem));
   assert(he != NULL);
   
   hcode                   = elem_hash(elem) % hash->size;
   he->value.elem_idx.elem = elem;
   he->value.elem_idx.idx  = idx;
   he->next                = hash->bucket[hcode];
   hash->bucket[hcode]     = he;
   hash->elems++;
}

/* Liefert -1 wenn nicht gefunden.
 */
int hash_lookup_elem_idx(Hash const* hash, Elem const* elem)
{
   unsigned int hcode = elem_hash(elem) % hash->size;
   HElem const* he;
   
   assert(hash_is_valid(hash));
   assert(elem_is_valid(elem));

   for(he = hash->bucket[hcode]; he != NULL; he = he->next)
      if (!elem_cmp(he->value.elem_idx.elem, elem))
         break;

   if (he == NULL)
      return -1;

   assert(he != NULL);

   return he->value.elem_idx.idx;
}

static void hash_statist(FILE* fp, Hash const* hash)
{
   HElem* he;
   int    min    = (int)hash->size;
   int    max    = 0;
   int    sum    = 0;
   int    zeros  = 0;
   int    filled = 0;
   double avg    = 0.0;
   unsigned int i;

   assert(fp != NULL);
   assert(hash_is_valid(hash));

   for(i = 0; i < hash->size; i++)
   {
      int count = 0;
      
      for(he = hash->bucket[i]; he != NULL; he = he->next)
         count++;

      if (count == 0)
         zeros++;
      else
         filled++;

      if (count < min)
         min = count;
      if (count > max)
         max = count;
      sum += count;
   }
   assert(sum == hash->elems);

   if (filled > 0)
      avg = (double)sum / (double)filled;
   
   fprintf(fp,
      "HashStat: size=%8u sum=%6d min=%3d max=%3d avg=%4.1f zero=%6d filled=%6d\n",
      hash->size, sum, min, max, avg, zeros, filled);
}


