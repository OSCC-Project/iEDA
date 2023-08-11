/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   gcgpqueue.c
 * @brief  methods for working with priority queue
 * @author Jonas Witt
 *
 * Various methods to work with the priority queue
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "gcgpqueue.h"
#include "pub_gcgpqueue.h"

#include "gcg.h"
#include "scip/def.h"
#include "scip/scip.h"
#include "blockmemshell/memory.h"

#include <assert.h>



/*
 * Priority Queue
 */

#define PQ_PARENT(q) (((q)+1)/2-1)
#define PQ_LEFTCHILD(p) (2*(p)+1)
#define PQ_RIGHTCHILD(p) (2*(p)+2)


/** resizes element memory to hold at least the given number of elements */
static
SCIP_RETCODE pqueueResize(
   GCG_PQUEUE*           pqueue,             /**< pointer to a priority queue */
   int                   minsize             /**< minimal number of storable elements */
   )
{
   int newsize;
   assert(pqueue != NULL);

   if( minsize <= pqueue->size )
      return SCIP_OKAY;

   newsize = SCIPcalcMemGrowSize(pqueue->scip, minsize);
   SCIP_CALL( SCIPreallocBlockMemoryArray(pqueue->scip, &pqueue->slots, pqueue->size, newsize) );
   pqueue->size = newsize;

   return SCIP_OKAY;
}

/** heapifies element at position pos into corresponding subtrees */
static
SCIP_RETCODE pqueueHeapify(
   GCG_PQUEUE*           pqueue,             /**< pointer to a priority queue */
   int                   pos                 /**< heapify element at position pos into corresponding subtrees */
   )
{
   int childpos;
   int brotherpos;
   void* elem;

   assert(pqueue != NULL);
   assert(pqueue->len > 0);

   /* move the better child of elem to its parents position until element
    * is better than its children
    */
   elem = pqueue->slots[pos];

   while( pos <= PQ_PARENT(pqueue->len-1) )
   {
      childpos = PQ_LEFTCHILD(pos);
      brotherpos = PQ_RIGHTCHILD(pos);
      if( brotherpos < pqueue->len && (*pqueue->ptrcomp)(pqueue->slots[brotherpos], pqueue->slots[childpos]) < 0 )
         childpos = brotherpos;
      if( (*pqueue->ptrcomp)(elem, pqueue->slots[childpos]) < 0 )
         break;
      pqueue->slots[pos] = pqueue->slots[childpos];
      pqueue->slots[childpos] = elem;

      pos = childpos;
   }
   assert(0 <= pos && pos < pqueue->len);

   return SCIP_OKAY;
}


/** creates priority queue */
SCIP_RETCODE GCGpqueueCreate(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_PQUEUE**          pqueue,             /**< pointer to a priority queue */
   int                   initsize,           /**< initial number of available element slots */
   SCIP_DECL_SORTPTRCOMP((*ptrcomp))         /**< data element comparator */
   )
{
   assert(pqueue != NULL);
   assert(ptrcomp != NULL);

   initsize = MAX(1, initsize);

   SCIP_CALL( SCIPallocBlockMemory(scip, pqueue) );
   (*pqueue)->len = 0;
   (*pqueue)->size = 0;
   (*pqueue)->scip = scip;
   (*pqueue)->slots = NULL;
   (*pqueue)->ptrcomp = ptrcomp;
   SCIP_CALL( pqueueResize(*pqueue, initsize) );

   return SCIP_OKAY;
}

/** frees priority queue, but not the data elements themselves */
void GCGpqueueFree(
   GCG_PQUEUE**         pqueue              /**< pointer to a priority queue */
   )
{
   assert(pqueue != NULL);

   SCIPfreeBlockMemoryArray((*pqueue)->scip, &(*pqueue)->slots, (*pqueue)->size);
   SCIPfreeBlockMemory((*pqueue)->scip, pqueue);
}

/** clears the priority queue, but doesn't free the data elements themselves */
void GCGpqueueClear(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   assert(pqueue != NULL);

   pqueue->len = 0;
}

/** inserts element into priority queue */
SCIP_RETCODE GCGpqueueInsert(
   GCG_PQUEUE*           pqueue,             /**< priority queue */
   void*                 elem                /**< element to be inserted */
   )
{
   int pos;
   int parentpos;

   assert(pqueue != NULL);
   assert(pqueue->len >= 0);
   assert(elem != NULL);

   SCIP_CALL( pqueueResize(pqueue, pqueue->len+1) );

   /* insert element as leaf in the tree, move it towards the root as long it is better than its parent */
   pos = pqueue->len;
   parentpos = PQ_PARENT(pos);
   pqueue->len++;
   while( pos > 0 && (*pqueue->ptrcomp)(elem, pqueue->slots[parentpos]) < 0 )
   {
      pqueue->slots[pos] = pqueue->slots[parentpos];
      pos = parentpos;
      parentpos = PQ_PARENT(pos);
   }
   pqueue->slots[pos] = elem;

   return SCIP_OKAY;
}

/** removes and returns best element from the priority queue */
void* GCGpqueueRemove(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   void* root;
   void* last;
   int pos;
   int childpos;
   int brotherpos;

   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   if( pqueue->len == 0 )
      return NULL;

   /* remove root element of the tree, move the better child to its parents position until the last element
    * of the queue could be placed in the empty slot
    */
   root = pqueue->slots[0];
   last = pqueue->slots[pqueue->len-1];
   pqueue->len--;
   pos = 0;
   while( pos <= PQ_PARENT(pqueue->len-1) )
   {
      childpos = PQ_LEFTCHILD(pos);
      brotherpos = PQ_RIGHTCHILD(pos);
      if( brotherpos <= pqueue->len && (*pqueue->ptrcomp)(pqueue->slots[brotherpos], pqueue->slots[childpos]) < 0 )
         childpos = brotherpos;
      if( (*pqueue->ptrcomp)(last, pqueue->slots[childpos]) <= 0 )
         break;
      pqueue->slots[pos] = pqueue->slots[childpos];
      pos = childpos;
   }
   assert(pos <= pqueue->len);
   pqueue->slots[pos] = last;

   return root;
}

/** resorts priority queue after changing the key values */
SCIP_RETCODE GCGpqueueResort(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   int lastinner;
   int pos;

   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   if( pqueue->len == 0 )
   {
      return SCIP_OKAY;
   }

   lastinner = PQ_PARENT(pqueue->len - 1);

   for( pos = lastinner; pos >= 0; --pos )
   {
      SCIP_CALL( pqueueHeapify(pqueue, pos) );
   }

   return SCIP_OKAY;
}


/** set the comperator of the priority queue */
SCIP_RETCODE GCGpqueueSetComperator(
   GCG_PQUEUE*           pqueue,             /**< priority queue */
   SCIP_DECL_SORTPTRCOMP((*ptrcomp))         /**< data element comparator */
   )
{
   pqueue->ptrcomp = ptrcomp;

   return SCIP_OKAY;
}

/* todo: write method to get comperator */

/**< delete item at position pos and insert last item at this position and resort pqueue */
SCIP_RETCODE GCGpqueueDelete(
   GCG_PQUEUE*          pqueue,             /**< priority queue */
   int                  pos,                /**< position of item that should be deleted */
   void**               elem                /**< pointer to store element that was deleted from pqueue */
   )
{
   void* last;
   int childpos;
   int brotherpos;

   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   assert(pos < pqueue->len);

   /* remove element at position pos of the tree, move the better child to its parents position until the last element
    * of the queue could be placed in the empty slot
    */
   *elem = pqueue->slots[pos];
   last = pqueue->slots[pqueue->len-1];
   pqueue->len--;
   while( pos <= PQ_PARENT(pqueue->len-1) )
   {
      childpos = PQ_LEFTCHILD(pos);
      brotherpos = PQ_RIGHTCHILD(pos);
      if( brotherpos <= pqueue->len && (*pqueue->ptrcomp)(pqueue->slots[brotherpos], pqueue->slots[childpos]) < 0 )
         childpos = brotherpos;
      if( (*pqueue->ptrcomp)(last, pqueue->slots[childpos]) <= 0 )
         break;
      pqueue->slots[pos] = pqueue->slots[childpos];
      pos = childpos;
   }
   assert(pos <= pqueue->len);
   pqueue->slots[pos] = last;

   return SCIP_OKAY;
}

/** returns the best element of the queue without removing it */
void* GCGpqueueFirst(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   if( pqueue->len == 0 )
      return NULL;

   return pqueue->slots[0];
}

/** returns the number of elements in the queue */
int GCGpqueueNElems(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   return pqueue->len;
}

/** returns the elements of the queue; changing the returned array may destroy the queue's ordering! */
void** GCGpqueueElems(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   )
{
   assert(pqueue != NULL);
   assert(pqueue->len >= 0);

   return pqueue->slots;
}
