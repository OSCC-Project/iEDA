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

/**@file   pricingprob.h
 * @brief  private methods for working with pricing problems, to be used by the pricing controller only
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef GCG_PRICINGPROB_H__
#define GCG_PRICINGPROB_H__

#include "struct_pricingprob.h"
#include "type_pricingprob.h"

#include "pricer_gcg.h"
#include "type_colpool.h"
#include "type_pricestore_gcg.h"
#include "type_pricingjob.h"
#include "type_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/** create a pricing problem */
SCIP_EXPORT
SCIP_RETCODE GCGpricingprobCreate(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGPROB**     pricingprob,        /**< pricing problem to be created */
   SCIP*                 pricingscip,        /**< SCIP data structure of the corresponding pricing problem */
   int                   probnr,             /**< index of the corresponding pricing problem */
   int                   nroundscol          /**< number of previous pricing rounds for which the number of improving columns should be counted */
);

/** free a pricing problem */
SCIP_EXPORT
void GCGpricingprobFree(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGPROB**     pricingprob         /**< pricing problem to be freed */
);

/** initialize pricing problem at the beginning of the pricing round */
SCIP_EXPORT
void GCGpricingprobInitPricing(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** uninitialize pricing problem at the beginning of the pricing round */
SCIP_EXPORT
void GCGpricingprobExitPricing(
   GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
   int                   nroundscol          /**< number of previous pricing rounds for which the number of improving columns should be counted */
   );

/** add generic branching data (constraint and dual value) to the current pricing problem */
SCIP_EXPORT
SCIP_RETCODE GCGpricingprobAddGenericBranchData(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
   SCIP_CONS*            branchcons,         /**< generic branching constraint */
   SCIP_Real             branchdual          /**< corresponding dual solution value */
   );

/** reset the pricing problem statistics for the current pricing round */
SCIP_EXPORT
void GCGpricingprobReset(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** update solution information of a pricing problem */
SCIP_EXPORT
void GCGpricingprobUpdate(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
   GCG_PRICINGSTATUS     status,             /**< status of last pricing job */
   SCIP_Real             lowerbound,         /**< new lower bound */
   int                   nimpcols            /**< number of new improving columns */
   );

/** add the information that the next branching constraint must be added */
SCIP_EXPORT
void GCGpricingprobNextBranchcons(
   GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
   );

/** set the lower bound of a pricing job */
SCIP_EXPORT
void GCGpricingjobSetLowerbound(
   GCG_PRICINGJOB*       pricingjob,         /**< pricing job */
   SCIP_Real             lowerbound          /**< new lower bound */
   );

#ifdef __cplusplus
}
#endif

#endif
