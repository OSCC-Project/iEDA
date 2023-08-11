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

/**@file   pricestore_gcg.h
 * @brief  methods for storing priced cols (based on SCIP's separation storage)
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GCG_PRICESTORE_H__
#define __GCG_PRICESTORE_H__


#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_implics.h"
#include "scip/type_retcode.h"
#include "scip/type_set.h"
#include "scip/type_stat.h"
#include "scip/type_event.h"
#include "scip/type_lp.h"
#include "scip/type_prob.h"
#include "scip/type_tree.h"
#include "scip/type_reopt.h"
#include "scip/type_branch.h"

#include "pub_colpool.h"
#include "pub_gcgcol.h"
#include "type_pricestore_gcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates price storage */
extern
SCIP_RETCODE GCGpricestoreCreate(
   SCIP*                 scip,                /**< SCIP data structure */
   GCG_PRICESTORE**      pricestore,          /**< pointer to store price storage */
   SCIP_Real             redcostfac,          /**< factor of -redcost/norm in score function */
   SCIP_Real             objparalfac,         /**< factor of objective parallelism in score function */
   SCIP_Real             orthofac,            /**< factor of orthogonalities in score function */
   SCIP_Real             mincolorth,          /**< minimal orthogonality of columns to add
                                                  (with respect to columns added in the current round) */
   GCG_EFFICIACYCHOICE   efficiacychoice      /**< choice to base efficiacy on */
   );

/** frees price storage */
extern
SCIP_RETCODE GCGpricestoreFree(
   SCIP*                 scip,                /**< SCIP data structure */
   GCG_PRICESTORE**      pricestore           /**< pointer to store price storage */
   );

/** informs price storage, that Farkas pricing starts now */
extern
void GCGpricestoreStartFarkas(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** informs price storage, that Farkas pricing is now finished */
extern
void GCGpricestoreEndFarkas(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** informs price storage, that the following cols should be used in any case */
extern
void GCGpricestoreStartForceCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** informs price storage, that the following cols should no longer be used in any case */
extern
void GCGpricestoreEndForceCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** adds col to price storage;
 *  if the col should be forced to enter the LP, an infinite score has to be used
 */
extern
SCIP_RETCODE GCGpricestoreAddCol(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COL*              col,                /**< priced col */
   SCIP_Bool             forcecol            /**< should the col be forced to enter the LP? */
   );

/** adds cols to priced vars and clears price storage */
extern
SCIP_RETCODE GCGpricestoreApplyCols(
   GCG_PRICESTORE*       pricestore,         /**< price storage */
   GCG_COLPOOL*          colpool,            /**< GCG column pool */
   SCIP_Bool             usecolpool,         /**< use column pool? */
   int*                  nfoundvars          /**< pointer to store number of variables that were added to the problem */
   );

/** clears the price storage without adding the cols to priced vars */
extern
void GCGpricestoreClearCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** removes cols that are inefficacious w.r.t. the current LP solution from price storage without adding the cols to the LP */
extern
void GCGpricestoreRemoveInefficaciousCols(
   GCG_PRICESTORE*       pricestore          /**< price storage */
   );

/** get cols in the price storage */
extern
GCG_COL** GCGpricestoreGetCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** get number of cols in the price storage */
extern
int GCGpricestoreGetNCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** get number of efficacious cols in the price storage */
extern
int GCGpricestoreGetNEfficaciousCols(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** get total number of cols found so far */
extern
int GCGpricestoreGetNColsFound(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** get number of cols found so far in current price round */
extern
int GCGpricestoreGetNColsFoundRound(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** get total number of cols applied to the LPs */
extern
int GCGpricestoreGetNColsApplied(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

/** gets time in seconds used for pricing cols from the pricestore */
extern
SCIP_Real GCGpricestoreGetTime(
   GCG_PRICESTORE*       pricestore           /**< price storage */
   );

#ifdef __cplusplus
}
#endif

#endif
