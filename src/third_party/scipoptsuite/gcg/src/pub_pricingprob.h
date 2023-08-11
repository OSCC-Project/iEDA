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

/**@file   pub_pricingprob.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for working with pricing problems
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef GCG_PUB_PRICINGPROB_H__
#define GCG_PUB_PRICINGPROB_H__

#include "type_pricingprob.h"
#include "scip/type_scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GCG Pricing Problem
 */

/**
 * @ingroup PRICINGPROB
 * @{
 */


/** get the SCIP instance corresponding to the pricing problem */
SCIP_EXPORT
SCIP* GCGpricingprobGetPricingscip(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the index of the corresponding pricing problem */
SCIP_EXPORT
int GCGpricingprobGetProbnr(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get generic branching data corresponding to the pricing problem */
SCIP_EXPORT
void GCGpricingprobGetGenericBranchData(
   GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
   SCIP_CONS***          branchconss,        /**< pointer to store branching constraints array, or NULL */
   SCIP_Real**           branchduals,        /**< pointer to store array of corresponding dual values, or NULL */
   int*                  nbranchconss        /**< pointer to store number of generic branching constraints, or NULL */
   );

/** get the number of generic branching constraints corresponding to the pricing problem */
SCIP_EXPORT
int GCGpricingprobGetNGenericBranchconss(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get index of current generic branching constraint considered the pricing problem */
SCIP_EXPORT
int GCGpricingprobGetBranchconsIdx(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** check if the current generic branching constraint has already been added */
SCIP_EXPORT
SCIP_Bool GCGpricingprobBranchconsIsAdded(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** mark the current generic branching constraint to be added */
SCIP_EXPORT
void GCGpricingprobMarkBranchconsAdded(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the status of a pricing problem */
SCIP_EXPORT
GCG_PRICINGSTATUS GCGpricingprobGetStatus(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the lower bound of a pricing problem */
SCIP_EXPORT
SCIP_Real GCGpricingprobGetLowerbound(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the number of improving columns found for this pricing problem */
SCIP_EXPORT
int GCGpricingprobGetNImpCols(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the number of times the pricing problem was solved during the loop */
SCIP_EXPORT
int GCGpricingprobGetNSolves(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** get the total number of improving colums found in the last pricing rounds */
SCIP_EXPORT
int GCGpricingprobGetNColsLastRounds(
   GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
   int                   nroundscol          /**< number of previous pricing rounds for which the number of improving columns should be counted */
   );

/**@} */

#ifdef __cplusplus
}
#endif
#endif
