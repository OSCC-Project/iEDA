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

/**@file   pub_gcgcol.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for working with gcg columns
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#ifndef GCG_PUB_GCGCOL_H__
#define GCG_PUB_GCGCOL_H__

#include "type_gcgcol.h"
#include "scip/type_scip.h"
#include "scip/type_retcode.h"
#include "scip/type_var.h"
#include "scip/type_cons.h"
#include "scip/type_misc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GCG Column
 */

/**@defgroup GCG_COLUMN GCG Column
 * @ingroup DATASTRUCTURES
 * @{
 */

/** create a gcg column */
extern
SCIP_RETCODE GCGcreateGcgCol(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL**            gcgcol,             /**< pointer to store gcg column */
   int                  prob,               /**< number of corresponding pricing problem */
   SCIP_VAR**           vars,               /**< (sorted) array of variables of corresponding pricing problem */
   SCIP_Real*           vals,               /**< array of solution values (belonging to vars) */
   int                  nvars,              /**< number of variables */
   SCIP_Bool            isray,              /**< is the column a ray? */
   SCIP_Real            redcost             /**< last known reduced cost */
);

/** free a gcg column */
extern
void GCGfreeGcgCol(
   GCG_COL**            gcgcol              /**< pointer to store gcg column */
);

/** create a gcg column from a solution to a pricing problem */
extern
SCIP_RETCODE GCGcreateGcgColFromSol(
   SCIP*                scip,               /**< SCIP data structure (original problem) */
   GCG_COL**            gcgcol,             /**< pointer to store gcg column */
   int                  prob,               /**< number of corresponding pricing problem */
   SCIP_SOL*            sol,                /**< solution of pricing problem with index prob */
   SCIP_Bool            isray,              /**< is column a ray? */
   SCIP_Real            redcost             /**< last known reduced cost */
);

/** get pricing problem index of gcg column */
extern
int GCGcolGetProbNr(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** get pricing problem of gcg column */
extern
SCIP* GCGcolGetPricingProb(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** get variables of gcg column */
extern
SCIP_VAR** GCGcolGetVars(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** get values of gcg column */
extern
SCIP_Real* GCGcolGetVals(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** get number of variables of gcg column */
extern
int GCGcolGetNVars(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** is gcg column a ray? */
extern
SCIP_Bool GCGcolIsRay(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** get reduced cost of gcg column */
extern
SCIP_Real GCGcolGetRedcost(
   GCG_COL*             gcgcol              /**< gcg column structure */
);

/** comparison method for sorting gcg columns by non-decreasing reduced cost */
extern
SCIP_DECL_SORTPTRCOMP(GCGcolCompRedcost);

/** comparison method for sorting gcg columns by non-increasing age */
extern
SCIP_DECL_SORTPTRCOMP(GCGcolCompAge);

/** comparison method for gcg columns. Returns TRUE iff columns are equal */
extern
SCIP_Bool GCGcolIsEq(
   GCG_COL*             gcgcol1,            /**< first gcg column structure */
   GCG_COL*             gcgcol2             /**< second gcg column structure */
);

/** update reduced cost of variable and increase age */
extern
void GCGcolUpdateRedcost(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real            redcost,            /**< new reduced cost */
   SCIP_Bool            growold             /**< increase age counter? */
   );

/** return solution value of variable in gcg column */
extern
SCIP_Real GCGcolGetSolVal(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol,             /**< gcg column */
   SCIP_VAR*            var                 /**< variable */
   );

/** get master coefficients of column */
extern
SCIP_Real* GCGcolGetMastercoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** get number of master coefficients of column */
extern
int GCGcolGetNMastercoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** set master coefficients of column */
extern
SCIP_RETCODE GCGcolSetMastercoefs(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real*           mastercoefs,        /**< array of master coefficients */
   int                  nmastercoefs        /**< number of master coefficients */
   );

/** set norm of column */
extern
void GCGcolSetNorm(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real            norm                /**< norm of column */
   );
/** get norm of column */
extern
void GCGcolComputeNorm(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** set master coefficients of column as initialized */
extern
SCIP_RETCODE GCGcolSetInitializedCoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** return if master coefficients of column have been initialized */
extern
SCIP_Bool GCGcolGetInitializedCoefs(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** get master coefficients of column */
extern
int* GCGcolGetLinkvars(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** get number of master coefficients of column */
extern
int GCGcolGetNLinkvars(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** set master coefficients information of column */
extern
SCIP_RETCODE GCGcolSetLinkvars(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   int*                 linkvars,           /**< array of linking variable indices for gcgcol->var */
   int                  nlinkvars           /**< number of linking variables in gcgcol->var */
   );

/** get master cut coefficients of column */
extern
SCIP_Real* GCGcolGetMastercuts(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** get number of master cut coefficients of column */
extern
int GCGcolGetNMastercuts(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** get norm of column */
extern
SCIP_Real GCGcolGetNorm(
   GCG_COL*             gcgcol              /**< gcg column structure */
   );

/** update master cut coefficients information of column */
extern
SCIP_RETCODE GCGcolUpdateMastercuts(
   GCG_COL*             gcgcol,             /**< gcg column structure */
   SCIP_Real*           newmastercuts,      /**< pointer to new array of master cut coefficients */
   int                  nnewmastercuts      /**< new number of master cut coefficients */
   );

/** gets the age of the col */
extern
int GCGcolGetAge(
   GCG_COL*             col                 /**< col */
   );

/** returns whether the col's age exceeds the age limit */
extern
SCIP_Bool GCGcolIsAged(
   GCG_COL*              col,                /**< col to check */
   int                   agelimit            /**< maximum age a col can reach before it is deleted from the pool, or -1 */
   );

/** compute parallelism of column to dual objective */
SCIP_Real GCGcolComputeDualObjPara(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol              /**< gcg column */
   );

/** compute orthogonality of two gcg columns */
extern
SCIP_Real GCGcolComputeOrth(
   SCIP*                scip,               /**< SCIP data structure */
   GCG_COL*             gcgcol1,            /**< first gcg column */
   GCG_COL*             gcgcol2             /**< second gcg column */
   );

/**@} */

#ifdef __cplusplus
}
#endif
#endif
