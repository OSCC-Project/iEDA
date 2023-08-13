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

/**@file   pub_colpool.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for storing cols in a col pool
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PUB_COLPOOL_H__
#define __SCIP_PUB_COLPOOL_H__


#include "scip/def.h"
#include "type_colpool.h"
#include "type_gcgcol.h"
#include "type_pricestore_gcg.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup DATASTRUCTURES
 * @{
 */

/** gets array of cols in the col pool */
SCIP_EXPORT
GCG_COL** GCGcolpoolGetCols(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** get number of cols in the col pool */
SCIP_EXPORT
int GCGcolpoolGetNCols(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** get maximum number of cols that were stored in the col pool at the same time */
SCIP_EXPORT
int GCGcolpoolGetMaxNCols(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** gets time in seconds used for pricing cols from the pool */
SCIP_EXPORT
SCIP_Real GCGcolpoolGetTime(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** get number of times, the col pool was separated */
SCIP_EXPORT
SCIP_Longint GCGcolpoolGetNCalls(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** get total number of cols that were priced from the col pool */
SCIP_EXPORT
SCIP_Longint GCGcolpoolGetNColsFound(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** creates col pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolCreate(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_COLPOOL**         colpool,            /**< pointer to store col pool */
   int                   agelimit            /**< maximum age a col can reach before it is deleted from the pool */
   );

/** frees col pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolFree(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COLPOOL**        colpool             /**< pointer to store col pool */
   );

/** removes all cols from the col pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolClear(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** if not already existing, adds col to col pool and captures it */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolAddCol(
   GCG_COLPOOL*          colpool,            /**< col pool */
   GCG_COL*              col,                /**< column to add */
   SCIP_Bool*            success             /**< pointer to store if col was added */
   );

/** adds col to col pool and captures it; doesn't check for multiple cols */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolAddNewCol(
   GCG_COLPOOL*         colpool,            /**< col pool */
   GCG_COL*             col                 /**< column to add */
   );

/** removes the col from the col pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolDelCol(
   GCG_COLPOOL*          colpool,            /**< col pool */
   GCG_COL*              col,                /**< col to remove */
   SCIP_Bool             freecol             /**< should the col be freed? */
   );

/** update node at which columns of column pool are feasible */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolUpdateNode(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** update reduced cost of columns in column pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolUpdateRedcost(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** gets number of cols in the col pool */
SCIP_EXPORT
void GCGcolpoolStartFarkas(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** gets number of cols in the col pool */
SCIP_EXPORT
void GCGcolpoolEndFarkas(
   GCG_COLPOOL*         colpool             /**< col pool */
   );

/** prices cols of the col pool */
SCIP_EXPORT
SCIP_RETCODE GCGcolpoolPrice(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_COLPOOL*          colpool,            /**< col pool */
   GCG_PRICESTORE*       pricestore,         /**< GCG price storage */
   SCIP_SOL*             sol,                /**< solution to be separated (or NULL for LP-solution) */
   SCIP_Bool*            foundvars           /**< pointer to store the result of the separation call */
   );

/** @} */

#ifdef __cplusplus
}
#endif

#endif
