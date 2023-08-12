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

/**@file   gcgsort.h
 * @brief  sorting functions, adapted from SCIP's sorttpl to include userdata
 * @author Tobias Oelschlegel
 **/

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_SORT_H__
#define GCG_SORT_H__

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** compares two element indices
 *  result:
 *    < 0: ind1 comes before (is better than) ind2
 *    = 0: both indices have the same value
 *    > 0: ind2 comes after (is worse than) ind2
 **/
#define GCG_DECL_SORTINDCOMP(x) int x (void* userdata, void* dataptr, int ind1, int ind2)

/** compares two data element pointers
 *  result:
 *    < 0: elem1 comes before (is better than) elem2
 *    = 0: both elements have the same value
 *    > 0: elem2 comes after (is worse than) elem2
 **/
#define GCG_DECL_SORTPTRCOMP(x) int x (void* userdata, void* elem1, void* elem2)

SCIP_EXPORT
void GCGsortPtr(
   void**                ptrarray,           /**< pointer array to be sorted */
   GCG_DECL_SORTPTRCOMP((*ptrcomp)),         /**< data element comparator */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   len                 /**< length of array */
   );

/** sort of two joint arrays of pointers/pointers, sorted by first array in non-decreasing order */
SCIP_EXPORT
void GCGsortPtrPtr(
   void**                ptrarray1,          /**< first pointer array to be sorted */
   void**                ptrarray2,          /**< second pointer array to be permuted in the same way */
   GCG_DECL_SORTPTRCOMP((*ptrcomp)),         /**< data element comparator */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   len                 /**< length of arrays */
   );

#ifdef __cplusplus
}
#endif

#endif
