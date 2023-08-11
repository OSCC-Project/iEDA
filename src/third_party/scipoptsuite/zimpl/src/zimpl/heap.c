/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: heap.c                                                        */
/*   Name....: Heap Functions                                                */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2006-2022 by Thorsten Koch <koch@zib.de>
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
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/heap.h"

#define HEAP_SID  0x48656170

enum heap_type
{
   HEAP_ERR = 0, HEAP_ENTRY = 1
};

typedef enum heap_type HeapType;

struct heap
{
   SID
   HeapType  type;
   int       size;   /**< Maximale Anzahl von Elementen im Heap              */
   int       used;   /**< Aktuelle Anzahl von Elementen im Heap              */
   HeapData* data;
   HeapCmp   data_cmp;
};

static void heap_print(FILE* fp, Heap const* heap)
{
   int i;
   
   for(i = 0; i < heap->used; i++)
   {
      fprintf(fp, "%3d ", i);
      switch(heap->type)
      {
      case HEAP_ENTRY :
         entry_print(fp, heap->data[i].entry);
         break;
      default :
         abort();
      }
      fprintf(fp, "\n");
   }
}

bool heap_is_valid(Heap const* heap)
{
   HeapData const* data;
   int       i;
   
   if (  heap           == NULL
      || heap->type     == HEAP_ERR
      || heap->data     == NULL
      || heap->data_cmp == NULL
      || heap->size     <= 0
      || heap->used     <  0
      || heap->used     >  heap->size)
      return false;

   data = heap->data;
   
   /* Heap property
    */
   for(i = 0; i < heap->used / 2; i++)
   {
      if ((*heap->data_cmp)(data[i], data[i + i]) > 0) //lint !e453
      {
         heap_print(stderr, heap); //lint !e453
         return false;
      }
      if (i + i + 1 < heap->used && (*heap->data_cmp)(data[i], data[i + i + 1]) > 0) //lint !e453
      {
         // heap_print(stderr, heap);
         return false;
      }
   }
   return true;
}

static Heap* heap_new(
   HeapType type,
   int      size,
   HeapCmp  data_cmp)
{
   Heap* heap = calloc(1, sizeof(*heap));

   assert(type     == HEAP_ENTRY);
   assert(size     >  0);
   assert(data_cmp != NULL);
   assert(heap     != NULL);

   heap->type     = type;
   heap->size     = size;
   heap->used     = 0;
   heap->data     = calloc((size_t)heap->size, sizeof(*heap->data));
   heap->data_cmp = data_cmp;
   
   SID_set(heap, HEAP_SID);
   assert(heap_is_valid(heap));
   
   return heap;
}

Heap* heap_new_entry(
   int     size,
   HeapCmp heap_entry_cmp)
{
   assert(size           >  0);
   assert(heap_entry_cmp != NULL);
   
   return heap_new(HEAP_ENTRY, size, heap_entry_cmp);
}

void heap_free(Heap* heap)
{
   int i;
   
   assert(heap_is_valid(heap));

   for(i = 0; i < heap->used; i++)
   {
      switch(heap->type)
      {
      case HEAP_ENTRY :
         entry_free(heap->data[i].entry);
         break;
      default :
         abort();
      }
   }
   free(heap->data);
   free(heap);
}

static void swap_entries(Heap const* heap, int i, int j)
{
   HeapData* data = heap->data;
   HeapData  t;

   t       = data[i];
   data[i] = data[j];
   data[j] = t;
}

/* Bewegt einen Eintrag weiter nach unten/hinten im Vektor
 * bis die Teilsortierung des Baumes wieder hergestellt ist.
 */
static void sift_down(
   Heap const* heap,
   int         current)
{
   HeapData* data = heap->data;
   int       child;

   /* Heap shift down
    * (Oberstes Element runter und korrigieren)
    */         
   child = current * 2;

   if (child + 1 < heap->used)
      if ((*heap->data_cmp)(data[child + 1], data[child]) < 0)
         child++;

   while(child < heap->used && (*heap->data_cmp)(data[current], data[child]) > 0)
   {
      swap_entries(heap, current, child);

      current = child;
      child  += child;
      
      if (child + 1 < heap->used)
         if ((*heap->data_cmp)(data[child + 1], data[child]) < 0)
            child++;
   }
}

/* Bewegt einen Eintrag weiter hoch/nach unten im Vektor
 * bis die Teilsortierung des Baumes wieder hergestellt ist.
 */
static void sift_up(
   Heap const* heap,
   int         current)
{
   HeapData* data   = heap->data;
   int       parent = current / 2;
   
   /* Heap sift up 
    */
   while(current > 0 && (*heap->data_cmp)(data[parent], data[current]) > 0)
   {
      swap_entries(heap, current, parent);
      current = parent;
      parent /= 2;
   }
}

/* Sortiert einen Eintrag in den Heap ein.
 */
void heap_push_entry(
   Heap*  heap,
   Entry* entry)
{
   assert(heap_is_valid(heap));
   assert(entry_is_valid(entry));
   assert(heap->used <  heap->size);

   heap->data[heap->used].entry = entry;
   
   heap->used++;

   sift_up(heap, heap->used - 1);

   assert(heap_is_valid(heap));
}

/* Holt den Eintrag mit dem kleinsten Wert aus dem Heap heraus.
 */
Entry* heap_pop_entry(
   Heap* heap)
{
   Entry* entry;
   
   assert(heap_is_valid(heap));
   assert(heap->used > 0);
   assert(heap->type == HEAP_ENTRY);
   
   /* Heap shift down
    * (Oberstes Element runter und korrigieren)
    */         
   entry = heap->data[0].entry;

   heap->data[0].entry = NULL;
   
   heap->used--;

   swap_entries(heap, 0, heap->used);
   
   sift_down(heap, 0);

   assert(heap_is_valid(heap));
      
   return entry;
}

Entry const* heap_top_entry(
   Heap const* heap)
{
   assert(heap_is_valid(heap));
   assert(heap->used > 0);
   assert(heap->type == HEAP_ENTRY);
   
   return heap->data[0].entry;
}

bool heap_is_full(Heap const* heap)
{
   assert(heap_is_valid(heap));

   return heap->used == heap->size;
}

bool heap_is_empty(Heap const* heap)
{
   assert(heap_is_valid(heap));

   return heap->used == 0;
}


