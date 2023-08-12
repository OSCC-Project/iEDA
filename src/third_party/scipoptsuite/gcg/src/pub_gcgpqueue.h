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

/**@file   pub_decomp.h
 * @ingroup PUBLICCOREAPI
 * @ingroup DATASTRUCTURES
 * @brief  public methods for working with priority queues
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef GCG_PUB_GCGPQUEUE_H__
#define GCG_PUB_GCGPQUEUE_H__

#include "scip/type_scip.h"
#include "scip/type_retcode.h"
#include "scip/type_var.h"
#include "scip/type_cons.h"
#include "scip/type_misc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Priority Queue
 */

/**@defgroup PriorityQueue Priority Queue
 * @ingroup DATASTRUCTURES
 * @{
 */

/** creates priority queue */
SCIP_EXPORT
SCIP_RETCODE GCGpqueueCreate(
   SCIP*                scip,                /** SCIP data structure */
   GCG_PQUEUE**         pqueue,              /**< pointer to a priority queue */
   int                   initsize,           /**< initial number of available element slots */
   SCIP_DECL_SORTPTRCOMP((*ptrcomp))         /**< data element comparator */
   );

/** frees priority queue, but not the data elements themselves */
SCIP_EXPORT
void GCGpqueueFree(
   GCG_PQUEUE**         pqueue              /**< pointer to a priority queue */
   );

/** clears the priority queue, but doesn't free the data elements themselves */
SCIP_EXPORT
void GCGpqueueClear(
   GCG_PQUEUE*          pqueue              /**< priority queue */
   );

/** inserts element into priority queue */
SCIP_EXPORT
SCIP_RETCODE GCGpqueueInsert(
   GCG_PQUEUE*          pqueue,             /**< priority queue */
   void*                 elem                /**< element to be inserted */
   );

/** removes and returns best element from the priority queue */
SCIP_EXPORT
void* GCGpqueueRemove(
   GCG_PQUEUE*          pqueue              /**< priority queue */
   );

/** resorts priority queue after changing the key values */
SCIP_EXPORT
SCIP_RETCODE GCGpqueueResort(
   GCG_PQUEUE*           pqueue              /**< priority queue */
   );

/** set the comperator of the priority queue */
SCIP_RETCODE GCGpqueueSetComperator(
   GCG_PQUEUE*           pqueue,             /**< priority queue */
   SCIP_DECL_SORTPTRCOMP((*ptrcomp))         /**< data element comparator */
   );

/**< delete item at position pos and insert last item at this position and resort pqueue */
extern
SCIP_RETCODE GCGpqueueDelete(
   GCG_PQUEUE*          pqueue,             /**< priority queue */
   int                  pos,                /**< position of item that should be deleted */
   void**               elem                /**< pointer to store element that was deleted from pqueue */
);

/** returns the best element of the queue without removing it */
SCIP_EXPORT
void* GCGpqueueFirst(
   GCG_PQUEUE*          pqueue              /**< priority queue */
   );

/** returns the number of elements in the queue */
SCIP_EXPORT
int GCGpqueueNElems(
   GCG_PQUEUE*          pqueue              /**< priority queue */
   );

/** returns the elements of the queue; changing the returned array may destroy the queue's ordering! */
SCIP_EXPORT
void** GCGpqueueElems(
   GCG_PQUEUE*          pqueue              /**< priority queue */
   );

/**@} */


#ifdef __cplusplus
}
#endif
#endif
