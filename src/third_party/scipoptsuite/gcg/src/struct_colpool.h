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

/**@file   struct_colpool.h
 * @ingroup DATASTRUCTURES
 * @brief  data structures for storing cols in a col pool
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_COLPOOL_H__
#define __SCIP_STRUCT_COLPOOL_H__


#include "scip/def.h"
#include "scip/type_clock.h"
#include "scip/type_misc.h"
#include "scip/type_lp.h"
#include "type_colpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/** storage for pooled cols */
struct GCG_Colpool
{
   SCIP*                 scip;               /**< SCIP data structure */
   SCIP_Longint          nodenr;             /**< node at which columns in colpool respect branching decisions */
   SCIP_Bool             infarkas;           /**< in Farkas pricing? */
   SCIP_Longint          ncalls;             /**< number of times, the colpool was separated */
   SCIP_Longint          ncolsfound;         /**< total number of cols that were separated from the pool */
   SCIP_CLOCK*           poolclock;          /**< pricing time */
   SCIP_HASHTABLE*       hashtable;          /**< hash table to identify already stored cols */
   GCG_COL**             cols;               /**< stored cols of the pool */
   SCIP_Longint          processedlp;        /**< last LP that has been processed for separating the LP */
   SCIP_Longint          processedlpsol;     /**< last LP that has been processed for separating other solutions */
   int                   colssize;           /**< size of cols array */
   int                   ncols;              /**< number of cols stored in the pool */
   int                   agelimit;           /**< maximum age a col can reach before it is deleted from the pool */
   int                   firstunprocessed;   /**< first col that has not been processed in the last LP */
   int                   firstunprocessedsol;/**< first col that has not been processed in the last LP when separating other solutions */
   int                   maxncols;           /**< maximal number of cols stored in the pool at the same time */
   SCIP_Bool             globalcolpool;      /**< is this the global col pool of SCIP? */
};

#ifdef __cplusplus
}
#endif

#endif
