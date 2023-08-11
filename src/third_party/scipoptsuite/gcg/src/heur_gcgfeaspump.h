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

/**@file   heur_gcgfeaspump.h
 * @ingroup PRIMALHEURISTICS
 * @brief  Objective Feasibility Pump 2.0
 * @author Timo Berthold
 * @author Domenico Salvagnin
 *
 * The fundamental idea of the Feasibility Pump is to construct two sequences of points which hopefully converge to a
 * feasible solution. One sequence consists of LP-feasiblepoints, the other one of integer feasible points.  They are
 * produced by alternately rounding an LP-feasible point and solvng an LP that finds a point on the LP polyhedron which
 * is closest to the rounded, integral point (w.r.t. Manhattan distance).
 *
 * The version implemented in SCIP supports using an Objective Feasibility Pump that uses a convex combination of the
 * Manhattan distance and the original LP objective for reoptimization. It further features Feasibility Pump 2.0
 * capabilities, hence propagating the fixings for a faster convergence.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_GCGFEASPUMP_H__
#define __SCIP_HEUR_GCGFEASPUMP_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the feaspump primal heuristic and includes it in SCIP */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeHeurGcgfeaspump(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
