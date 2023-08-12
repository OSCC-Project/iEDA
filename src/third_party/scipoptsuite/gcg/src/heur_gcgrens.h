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

/**@file   heur_gcgrens.h
 * @brief  LNS heuristic that finds the optimal rounding to a given point
 * @author Timo Berthold
 * @author Christian Puchert
 * @ingroup PRIMALHEURISTICS
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_HEUR_GCGRENS_H__
#define GCG_HEUR_GCGRENS_H__

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates RENS primal heuristic and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeHeurGcgrens(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** main procedure of the GCG RENS heuristic, creates and solves a sub-SCIP */
SCIP_RETCODE GCGapplyGcgrens(
   SCIP*                 scip,               /**< original SCIP data structure                                   */
   SCIP_HEUR*            heur,               /**< heuristic data structure                                       */
   SCIP_RESULT*          result,             /**< result data structure                                          */
   SCIP_Real             minfixingrate,      /**< minimum percentage of integer variables that have to be fixed  */
   SCIP_Real             minimprove,         /**< factor by which RENS should at least improve the incumbent     */
   SCIP_Longint          maxnodes,           /**< maximum number of  nodes for the subproblem                    */
   SCIP_Longint          nstallnodes,        /**< number of stalling nodes for the subproblem                    */
   SCIP_Bool             binarybounds,       /**< should general integers get binary bounds [floor(.),ceil(.)]?  */
   SCIP_Bool             uselprows           /**< should subproblem be created out of the rows in the LP rows?   */
   );

#ifdef __cplusplus
}
#endif

#endif
