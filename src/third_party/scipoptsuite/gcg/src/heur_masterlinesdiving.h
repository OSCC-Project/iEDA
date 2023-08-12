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

/**@file   heur_masterlinesdiving.h
 * @ingroup DIVINGHEURISTICS
 * @brief  master LP diving heuristic that fixes variables with a large difference to their root solution
 * @author Tobias Achterberg
 *
 * Diving heuristic: Iteratively fixes some fractional variable and resolves the LP-relaxation, thereby simulating a
 * depth-first-search in the tree. Line search diving chooses the variable with the greatest difference of its root LP
 * solution and the current LP solution, hence, the variable that developed most.  It is fixed to the next integer in
 * the direction it developed. One-level backtracking is applied: If the LP gets infeasible, the last fixing is undone,
 * and the opposite fixing is tried. If this is infeasible, too, the procedure aborts.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_MASTERLINESDIVING_H__
#define __SCIP_HEUR_MASTERLINESDIVING_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the masterlinesdiving primal heuristic and includes it in GCG */
SCIP_EXPORT
SCIP_RETCODE GCGincludeHeurMasterlinesdiving(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
